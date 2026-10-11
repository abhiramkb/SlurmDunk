"""Checks run before submission: duplicate jobs in the queue, and the executable's --help."""
import sys

import pytest

from slurmdunk import core
from slurmdunk.config_schema import validate_config

from conftest import SLURM, dedent, generate, write_config

# --------------------------------------------------------------------------- duplicate check


@pytest.fixture
def squeue(tmp_path, monkeypatch):
    """A stub squeue on PATH printing $STUB_OUT and exiting with $STUB_RC."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "squeue"
    stub.write_text('#!/bin/bash\n[ -n "$STUB_OUT" ] && cat "$STUB_OUT"\n'
                    '[ -n "$STUB_ERR" ] && echo "$STUB_ERR" >&2\nexit ${STUB_RC:-0}\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    return tmp_path


def test_duplicate_check_matches_new_and_legacy_tags(squeue, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")
    path = cfg.resolve()
    out = tmp_path / "queue.txt"
    out.write_text(dedent(f"""
        2014797|RUNNING|(null)
        3000001|PENDING|slurmdunk-config:{path}
        3000001|RUNNING|slurmdunk-config:{path}
        3000002|RUNNING|orchestrator-config:{path}
        3000003|PENDING|slurmdunk-config:{tmp_path}/other.yaml
        3000004|PENDING|weird|comment|with pipes
        3000005|PENDING|slurmdunk-config:{path}.bak
        """))
    monkeypatch.setenv("STUB_OUT", str(out))
    jobs, error = core.find_live_jobs_for_config(cfg)
    assert error is None
    assert jobs == {"3000001": {"PENDING", "RUNNING"}, "3000002": {"RUNNING"}}
    assert core.check_duplicate_submission(cfg) is True  # skip this config


def test_duplicate_check_failing_or_missing_squeue(squeue, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")
    monkeypatch.setenv("STUB_RC", "1")
    monkeypatch.setenv("STUB_ERR", "slurm_load_jobs error: Socket timed out")
    assert core.find_live_jobs_for_config(cfg)[1] == "slurm_load_jobs error: Socket timed out"
    assert core.check_duplicate_submission(cfg) is True  # cannot verify: skip
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert core.find_live_jobs_for_config(cfg) == (None, None)
    assert core.check_duplicate_submission(cfg) is False  # off the cluster: warn and continue


def test_generated_script_carries_the_new_tag(tmp_path, stub):
    cfg = write_config(tmp_path / "t.yaml", {"loopQ": False, "execution": {"executable": str(stub)}, "slurm": SLURM})
    script, _ = generate(cfg, tmp_path / "gen")
    assert f'#SBATCH --comment="slurmdunk-config:{cfg.resolve()}"' in script.read_text()


# --------------------------------------------------------------------------- --help check

FLAGS_PROG = """
import argparse
p = argparse.ArgumentParser()
p.add_argument("-x", type=float)
p.add_argument("--beta", type=float)
p.add_argument("--batch_size", type=int)
p.add_argument("--q")
p.add_argument("-v", "--verbose", action="store_true")
p.parse_args()
"""
POSITIONAL_PROG = """
import argparse
p = argparse.ArgumentParser()
p.add_argument("infile")
p.add_argument("outfile")
p.add_argument("--rtol", type=float)
p.parse_args()
"""


def help_check(tmp_path, execution, args, positional=()):
    cfg = {"execution": execution, "slurm": SLURM, "args": args, "positional_args": list(positional)}
    return core.validate_script_args(validate_config(cfg, "test", True))


@pytest.fixture
def programs(tmp_path):
    (tmp_path / "flags.py").write_text(FLAGS_PROG)
    (tmp_path / "pos.py").write_text(POSITIONAL_PROG)
    marker = tmp_path / "WAS_RUN"
    (tmp_path / "touch.sh").write_text(f"#!/bin/bash\ntouch {marker}\n")
    (tmp_path / "touch.sh").chmod(0o755)
    shebang = tmp_path / "flags_exec.py"
    shebang.write_text(f"#!{sys.executable}\n" + FLAGS_PROG)
    shebang.chmod(0o755)
    return tmp_path


@pytest.mark.parametrize("args, ok, flagged", [
    ({"x": 1, "beta": 2, "v": True, "verbose": True}, True, []),               # -x and -v as passed
    ({"batch-size": 4}, True, []),                                              # matches --batch_size
    ({"q": "a"}, False, ["-q"]),                                                # passed as -q; only --q exists
])
def test_help_check_flags(programs, args, ok, flagged):
    assert help_check(programs, {"interpreter": sys.executable, "executable": str(programs / "flags.py")}, args) \
        == (ok, flagged)


def test_help_check_positional_count(programs, capsys):
    pos = {"interpreter": sys.executable, "executable": str(programs / "pos.py")}
    assert help_check(programs, pos, {"infile": "a", "outfile": "b", "rtol": 1e-3}, ["infile", "outfile"]) == (True, [])
    assert "WARNING" not in capsys.readouterr().err
    assert help_check(programs, pos, {"infile": "a", "outfile": "b", "n": 3, "rtol": 1e-3},
                      ["infile", "outfile", "n"]) == (True, [])
    err = capsys.readouterr().err
    assert "lists 2 positional" in err and "has 3" in err


def test_help_check_never_runs_without_flags(programs):
    assert help_check(programs, {"executable": str(programs / "touch.sh")}, {"a": 1, "b": 2}, ["a", "b"]) == (True, [])
    assert not (programs / "WAS_RUN").exists()


def test_help_check_without_interpreter(programs):
    assert help_check(programs, {"executable": str(programs / "flags_exec.py")}, {"q": "a"}) == (False, ["-q"])


@pytest.mark.parametrize("usage, count", [
    ("usage: prog.py [-h] [--rtol RTOL] infile outfile\n", 2),
    ("usage: photon_T.jl [--neval NEVAL] [-h] Q beta x xmax\n", 4),
    ("usage: prog [-h] [--a A]\n            [--b B] infile\n\noptional arguments:\n", 1),
    ("usage: prog [-h] --x X infile\n", 1),                 # a required option's value is not counted
    ("usage: prog [-h] [--v [V ...]] files [files ...]\n", 1),
    ("no usage line here\n", None),
])
def test_count_usage_positionals(usage, count):
    assert core.count_usage_positionals(usage) == count
