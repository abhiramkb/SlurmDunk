"""Generate -> run the script under bash with a stub executable -> check what the
program received, what was logged, the checkpoints, and what 'collect' finds."""
import itertools
import os
import re
import sqlite3

import pytest

from conftest import SLURM, experiment, generate, needs_bash, parse_log, run_script, slurmdunk, write_config

pytestmark = needs_bash

# --------------------------------------------------------------------------- flags


FLAG_ARGS = {
    "a": 5,            # single letter            -> -a 5
    "n": -3,           # single letter, negative  -> -n -3
    "v": True,         # single-letter switch     -> -v
    "beta": -0.5,      # multi-letter, negative   -> --beta -0.5
    "verbose": True,   # multi-letter switch      -> --verbose
    "mode": "fast",
    "--Q": 2,          # explicit double dash
    "-nthreads": 4,    # explicit single dash, multi-letter
    "lo": "-inf",
    "quiet": True,     # switch directly followed by a single-letter flag
    "z": 1,
}
FLAG_ARGV = ["-a", "5", "-n", "-3", "-v", "--beta", "-0.5", "--verbose", "--mode", "fast", "--Q", "2",
             "-nthreads", "4", "--lo", "-inf", "--quiet", "-z", "1"]
FLAG_LOGGED = {"a": "5", "n": "-3", "v": "", "beta": "-0.5", "verbose": "", "mode": "fast", "Q": "2",
               "nthreads": "4", "lo": "-inf", "quiet": "", "z": "1"}


@pytest.mark.parametrize("array", [False, True], ids=["inner", "array"])
def test_every_flag_is_passed_logged_and_collected(tmp_path, stub, array):
    root = tmp_path / "db"
    cfg = {
        "execution": {"executable": str(stub)},
        "slurm": SLURM,
        "experiment": experiment(root, "flags", tracking_args={"save_dir": "{save_dir}_{__indicator}"}),
        "args": FLAG_ARGS,
        "inner_loop": {"type": "explicit", "arg_name": "x", "values": [1, -2]},
    }
    if array:
        cfg["outer_loops"] = [{"type": "explicit", "arg_name": "k", "values": [7]}]
    script, out = generate(write_config(tmp_path / "flags.yaml", cfg), tmp_path / "gen")
    assert script, out
    log = root / "flags" / "SLURM_OUTPUT" / ("flags_4242_0.out" if array else "flags_4243.out")
    assert run_script(script, log, tmp_path) == 0
    argv, logged = parse_log(log)

    job = "4242" if array else "4243"
    for line, x in ((1, "1"), (2, "-2")):
        ind = f"A0_L{line}" if array else f"L{line}"
        save = f"{root}/flags/{job}_{ind}"
        outer = ["-k", "7"] if array else []
        assert argv[ind] == FLAG_ARGV + outer + ["-x", x, "--save_dir", save]
        assert logged[ind] == {**FLAG_LOGGED, **({"k": "7"} if array else {}), "x": x, "save_dir": save}

    slurmdunk("collect", tmp_path / "flags.yaml", cwd=tmp_path, check=True)
    con = sqlite3.connect(root / "flags" / "results.db")
    cols = [c[1] for c in con.execute("pragma table_info(runs)")]
    assert set(FLAG_LOGGED) | {"x"} <= set(cols)


# --------------------------------------------------------------------------- executable


@pytest.mark.parametrize("mode", ["single", "inner", "array_concurrent", "bare_command"])
def test_executable_is_logged_and_collected(tmp_path, stub, mode):
    """Every run logs '[<run>_exec] path: ...', with git provenance off too, and collect makes it
    the 'executable' column. A bare command is resolved on PATH when the job runs."""
    root = tmp_path / "db"
    executable, expected, env = str(stub), str(stub.resolve()), {}
    if mode == "bare_command":
        bindir = tmp_path / "bin"
        bindir.mkdir()
        (bindir / "mytool").write_text(stub.read_text())
        (bindir / "mytool").chmod(0o755)
        executable, expected = "mytool", str(bindir / "mytool")
        env = {"PATH": f"{bindir}:{os.environ['PATH']}"}
    cfg = {
        "execution": {"executable": executable, "git_provenance": False},
        "slurm": SLURM,
        "experiment": experiment(root, "ex"),
        "args": {"n": 1},
    }
    if mode != "single":
        cfg["inner_loop"] = {"type": "explicit", "arg_name": "x", "values": [1, 2]}
    if mode == "array_concurrent":
        cfg["outer_loops"] = [{"type": "explicit", "arg_name": "k", "values": [7, 8]}]
        cfg["execution"]["multithreading_level"] = 1
        cfg["slurm"] = {**SLURM, "num_array_jobs": 1, "cpus-per-task": 2}
    config = write_config(tmp_path / "ex.yaml", cfg)
    script, out = generate(config, tmp_path / "gen")
    assert script, out
    log = root / "ex" / "SLURM_OUTPUT" / ("ex_4242_0.out" if mode == "array_concurrent" else "ex_4243.out")
    assert run_script(script, log, tmp_path, **env) == 0

    runs = {"single": ["SINGLE"], "inner": ["L1", "L2"], "bare_command": ["L1", "L2"],
            "array_concurrent": ["A0_L1", "A0_L2", "A1_L1", "A1_L2"]}[mode]
    assert dict(re.findall(r"^\[(\w+?)_exec\] path: (.*)$", log.read_text(), re.M)) == \
        {run: expected for run in runs}
    assert set(parse_log(log)[0]) == set(runs)  # the program ran for every run

    slurmdunk("collect", config, cwd=tmp_path, check=True)
    con = sqlite3.connect(root / "ex" / "results.db")
    assert [c[1] for c in con.execute("pragma table_info(runs)")][7] == "executable"
    assert dict(con.execute("select indicator, executable from runs")) == {run: expected for run in runs}


# --------------------------------------------------------------------------- positional


def positional_array_config(root, stub):
    """Array mode, packed into one element, concurrent inner loop; positionals from
    every source, ordered inner before outer, with awkward values."""
    return {
        "execution": {"executable": str(stub), "multithreading_level": 1},
        "slurm": {**SLURM, "num_array_jobs": 1, "cpus-per-task": 2},
        "experiment": experiment(root, "posA", tracking_args={
            "resfile": "{save_dir}_{__indicator}/res.json",    # positional
            "save_dir": "{save_dir}_{__indicator}",            # stays a flag
        }),
        "args": {"repeats": 5, "verbose": True, "label": "two words", "tag": 'a$b"c'},
        "outer_loops": [
            {"type": "explicit", "arg_name": "infile", "values": ["/data/a b.dat", "/data/c.dat"]},
            {"type": "explicit", "arg_name": "shift", "values": [-1, 2]},
        ],
        "inner_loop": {"type": "explicit", "arg_name": "rtol", "values": ["1e-6", "-3.5"]},
        "positional_args": ["rtol", "infile", "resfile", "repeats", "label", "tag"],
        "collect": {"json": {"path": "{resfile}"}},
    }


def test_positional_array_packed_concurrent(tmp_path, stub):
    root = tmp_path / "db"
    cfg = positional_array_config(root, stub)
    config = write_config(tmp_path / "A.yaml", cfg)
    script, out = generate(config, tmp_path / "gen")
    assert script, out
    log = root / "posA" / "SLURM_OUTPUT" / "posA_4242_0.out"
    assert run_script(script, log, tmp_path) == 0
    argv, logged = parse_log(log)

    combos = list(itertools.product(["/data/a b.dat", "/data/c.dat"], [-1, 2]))
    for o, (infile, shift) in enumerate(combos):
        for line, rtol in enumerate(["1e-6", "-3.5"], start=1):
            ind = f"A{o}_L{line}"
            save = f"{root}/posA/4242_{ind}"
            assert argv[ind] == ["--verbose", "--shift", str(shift), "--save_dir", save,
                                 rtol, infile, f"{save}/res.json", "5", "two words", 'a$b"c']
            assert logged[ind] == {"verbose": "", "shift": str(shift), "save_dir": save, "rtol": rtol,
                                   "infile": infile, "resfile": f"{save}/res.json", "repeats": "5",
                                   "label": "two words", "tag": 'a$b"c'}
    assert len(argv) == 8

    slurmdunk("collect", config, cwd=tmp_path, check=True)
    con = sqlite3.connect(root / "posA" / "results.db")
    rows = {r[0]: r for r in con.execute("select job_id, rtol, infile, tag, label, result from runs")}
    assert rows["4242_A2_L2"][1:5] == (-3.5, "/data/c.dat", 'a$b"c', "two words")
    assert len(rows) == 8 and {r[5] for r in rows.values()} == {11}  # JSON found via json.path '{resfile}'

    # Resuming: the same script skips everything; regenerating finds nothing pending
    rerun = log.with_name("rerun.out")
    run_script(script, rerun, tmp_path)
    text = rerun.read_text()
    assert text.count("[CHECKPOINT] Skipping") == 8 and "ARGV" not in text
    script2, out = generate(config, tmp_path / "gen")
    assert script2 is None and "complete" in out.lower()

    # Changing one outer positional value makes exactly its combinations pending
    cfg["outer_loops"][0]["values"] = ["/data/a b.dat", "/data/d.dat"]
    script3, _ = generate(write_config(config, cfg), tmp_path / "gen")
    members = re.search(r'^\s+\[0\]="([^"]*)"', script3.read_text(), re.M).group(1)
    assert members == "2 3"


def test_positional_first_inner_loop(tmp_path, stub):
    root = tmp_path / "db"
    cfg = {
        "execution": {"executable": str(stub)},
        "slurm": SLURM,
        "experiment": experiment(root, "posB"),
        "args": {"mode": "fast"},
        "inner_loop": {"type": "explicit", "arg_name": "x", "values": [0.1, -0.2]},
        "positional_args": ["x"],
        "positional_placement": "first",
    }
    config = write_config(tmp_path / "B.yaml", cfg)
    script, out = generate(config, tmp_path / "gen")
    log = root / "posB" / "SLURM_OUTPUT" / "posB_4243.out"
    assert run_script(script, log, tmp_path) == 0
    argv, logged = parse_log(log)
    assert argv == {"L1": ["0.1", "--mode", "fast"], "L2": ["-0.2", "--mode", "fast"]}
    assert logged == {"L1": {"mode": "fast", "x": "0.1"}, "L2": {"mode": "fast", "x": "-0.2"}}

    assert generate(config, tmp_path / "gen")[0] is None
    cfg["inner_loop"]["values"] = [0.1, -0.25]
    script3, _ = generate(write_config(config, cfg), tmp_path / "gen")
    assert "Pending Tasks: 1 / 2" in script3.read_text()


def test_positional_single_run_without_interpreter(tmp_path, stub):
    root = tmp_path / "db"
    cfg = {
        "loopQ": False,
        "execution": {"executable": str(stub)},
        "slurm": SLURM,
        "experiment": experiment(root, "posC", tracking_args={"outfile": "{save_dir}/out.json"}),
        "args": {"infile": "in.dat", "n": 3},
        "positional_args": ["infile", "outfile"],
    }
    config = write_config(tmp_path / "C.yaml", cfg)
    script, out = generate(config, tmp_path / "gen")
    log = root / "posC" / "SLURM_OUTPUT" / "posC_4243.out"
    assert run_script(script, log, tmp_path) == 0
    argv, logged = parse_log(log)
    out_json = f"{root}/posC/4243/out.json"
    assert argv == {"SINGLE": ["-n", "3", "in.dat", out_json]}
    assert logged == {"SINGLE": {"n": "3", "infile": "in.dat", "outfile": out_json}}

    script2, out = generate(config, tmp_path / "gen")
    assert script2 is None and "already completed" in out
    cfg["args"]["infile"] = "in2.dat"
    assert generate(write_config(config, cfg), tmp_path / "gen")[0] is not None


def test_bad_positional_loop_value_is_a_clean_error(tmp_path, stub):
    cfg = {
        "execution": {"executable": str(stub)},
        "slurm": SLURM,
        "inner_loop": {"type": "explicit", "arg_name": "x", "values": [1, True]},
        "positional_args": ["x"],
    }
    script, out = generate(write_config(tmp_path / "bad.yaml", cfg), tmp_path / "gen")
    assert script is None
    assert "positional argument 'x' is true/false" in out and "inner loop row 2" in out
    assert "Traceback" not in out
