"""Golden checkpoint names.

A checkpoint file is named after an md5 of the run's signature. If that
signature ever changes, every existing campaign silently restarts from scratch,
so the names below are pinned. They were computed with the code these campaigns
were run with (Code-Orchestrator 85f7159); the flags-only names also equal the
formula used before positional arguments existed (a02a3f9). Do not update these
values to make a test pass: a change here needs a migration path for old runs.
"""
import hashlib
import re
from pathlib import Path

import pytest

from slurmdunk.checkpoint_utils import filter_inner_loop_tasks, get_pending_array_task_ids

from conftest import SLURM, experiment, generate, needs_bash, run_script, write_config

EXEC = "python3 /opt/prog.py --mode fast"
INNER = [{"x": 0.1, "--Q": 2}, {"x": -0.2, "--Q": 3}]
OUTER = [{"dipole": "/d/a.dat", "Csq": -1}, {"dipole": "/d/b.dat", "Csq": 2}]

GOLDEN = {
    "inner_flags": ([], [
        "L1_9797b3c129291378026f3f62e6cf5f32.done",
        "L2_0b97cde56d0f78605bc6ef8617b72e8b.done",
    ]),
    "array_flags": ([], [
        "A0_L1_9d3789f545afb6735b035ee58247d237.done",
        "A0_L2_1176a398dd19210c6b52229a30ccb183.done",
        "A1_L1_b0db466e051f45fe7c6f3742a6317c39.done",
        "A1_L2_409d7b55abdfbc681f9c1753c82d2966.done",
    ]),
    "inner_positional": ([("x", "inner")], [
        "L1_81097e195c3f2efef58062ed54e366f5.done",
        "L2_c1c56e66eebfbb9e2c601eadc57f3ae8.done",
    ]),
    "array_positional": ([("x", "inner"), ("dipole", "outer")], [
        "A0_L1_5f54c8922c067cd987982464f9225c3c.done",
        "A0_L2_2003b41cc89dda24c30d1b9b1728644f.done",
        "A1_L1_ba94cc295638c7c1d7d3cc25c928aad9.done",
        "A1_L2_c99de6c40cb967e767eaa967187c9596.done",
    ]),
}


def pending(case, checkpoint_dir):
    positional, _ = GOLDEN[case]
    if case.startswith("inner"):
        return filter_inner_loop_tasks(checkpoint_dir, EXEC, INNER, positional)
    return get_pending_array_task_ids(checkpoint_dir, EXEC, OUTER, INNER, positional)


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_golden_names_count_as_done(tmp_path, case):
    assert pending(case, tmp_path) != []  # nothing done yet
    for name in GOLDEN[case][1]:
        (tmp_path / name).touch()
    assert pending(case, tmp_path) == []


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_each_golden_name_is_needed(tmp_path, case):
    """Leaving out any one name leaves work pending, so all of them are pinned."""
    names = GOLDEN[case][1]
    for skip in names:
        for name in names:
            if name != skip:
                (tmp_path / name).touch()
        assert pending(case, tmp_path) != []
        for f in tmp_path.iterdir():
            f.unlink()


def test_single_run_signature():
    assert hashlib.md5(EXEC.encode()).hexdigest() == "26308bc3ceeea51e3f491fab53abcbf3"
    assert hashlib.md5((EXEC + " in.dat").encode()).hexdigest() == "e3220862fd384cb153705ad6c9d5653d"


def test_exec_sig_composition(tmp_path):
    """EXEC_SIG is '<interpreter> <exec flags> <resolved executable> <fixed flags>', then any
    fixed positional values; the golden names above assume exactly this."""
    prog = tmp_path / "prog.py"
    prog.write_text("")
    base = {"execution": {"interpreter": "python3", "flags": ["-u"], "executable": str(prog)}, "slurm": SLURM,
            "loopQ": False}
    script, _ = generate(write_config(tmp_path / "a.yaml", {**base, "args": {"mode": "fast", "x": 1}}), tmp_path / "g1")
    assert f'EXEC_SIG="python3 -u {prog.resolve()} --mode fast -x 1"' in script.read_text()
    script, _ = generate(write_config(tmp_path / "b.yaml", {**base, "args": {"mode": "fast", "infile": 'a b$"'},
                                                           "positional_args": ["infile"]}), tmp_path / "g2")
    # the fixed positional value is appended, escaped so that bash recovers it exactly
    assert f'EXEC_SIG="python3 -u {prog.resolve()} --mode fast a b\\$\\""' in script.read_text()


@needs_bash
def test_runtime_hashes_match_the_python_prefilter(tmp_path, stub):
    """The bash runtime names its checkpoint files exactly as the Python pre-filter expects."""
    root = tmp_path / "db"
    cfg = {
        "execution": {"executable": str(stub)},
        "slurm": SLURM,
        "experiment": experiment(root, "h"),
        "args": {"mode": "fast"},
        "outer_loops": [{"type": "explicit", "arg_name": "dipole", "values": ["/d/a.dat", "/d/b.dat"]}],
        "inner_loop": {"type": "explicit", "arg_name": "x", "values": [0.1, -0.2]},
        "positional_args": ["x", "dipole"],
    }
    config = write_config(tmp_path / "h.yaml", cfg)
    script, _ = generate(config, tmp_path / "gen")
    for task in ("0", "1"):
        run_script(script, root / "h" / "SLURM_OUTPUT" / f"h_4242_{task}.out", tmp_path, SLURM_ARRAY_TASK_ID=task)
    written = sorted(p.name for p in (root / "h" / "checkpoints").iterdir())
    exec_sig = re.search(r'^EXEC_SIG="(.*)"$', script.read_text(), re.M).group(1)
    outer = [{"dipole": "/d/a.dat"}, {"dipole": "/d/b.dat"}]
    inner = [{"x": 0.1}, {"x": -0.2}]
    # The pre-filter, given what the runtime wrote, must find nothing pending
    pending_ids = get_pending_array_task_ids(Path(root / "h" / "checkpoints"), exec_sig, outer, inner,
                                             [("x", "inner"), ("dipole", "outer")])
    assert len(written) == 4 and pending_ids == []
