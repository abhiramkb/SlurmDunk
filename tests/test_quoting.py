"""Flag values are shell-quoted: spaces, quotes, $ and backticks reach the program
intact and literal, are logged exactly, and resume correctly."""
import sqlite3

import pytest

from slurmdunk.utils import format_cli_args

from conftest import SLURM, experiment, generate, needs_bash, parse_log, run_script, slurmdunk, write_config


def test_safe_values_are_unchanged():
    # Strings (and hence checkpoint hashes) of ordinary values are what they were before quoting
    args = {"path": "/p/mvgam,balsd/85.dat", "x": -0.5, "rtol": "1e-6", "n": 3, "on": True, "off": False, "none": None}
    assert format_cli_args(args) == "--path /p/mvgam,balsd/85.dat -x -0.5 --rtol 1e-6 -n 3 --on"


def test_unsafe_values_are_quoted():
    assert format_cli_args({"title": "two words", "v": "a$b", "q": "it's"}) == \
        "--title 'two words' -v 'a$b' -q 'it'\"'\"'s'"


AWKWARD = {"title": "two words", "quote": "it's \"q\"", "dollar": "$HOME", "tick": "`date`"}


@needs_bash
@pytest.mark.parametrize("array", [False, True], ids=["inner", "array"])
def test_awkward_flag_values_arrive_literally(tmp_path, stub, array):
    root = tmp_path / "db"
    cfg = {
        "execution": {"executable": str(stub)},
        "slurm": SLURM,
        "experiment": experiment(root, "q"),
        "args": AWKWARD,
        "inner_loop": {"type": "explicit", "arg_name": "label", "values": ["a b", "c'd"]},
    }
    if array:
        cfg["outer_loops"] = [{"type": "explicit", "arg_name": "tag", "values": ["$x y", "plain"]}]
    config = write_config(tmp_path / "q.yaml", cfg)
    script, out = generate(config, tmp_path / "gen")
    assert script, out
    logs = root / "q" / "SLURM_OUTPUT"
    log = logs / ("q_4242_0.out" if array else "q_4243.out")
    assert run_script(script, log, tmp_path) == 0
    if array:  # the second outer value runs as array task 1
        assert run_script(script, logs / "q_4242_1.out", tmp_path, SLURM_ARRAY_TASK_ID="1") == 0
    argv, logged = parse_log(log)

    fixed = ["--title", "two words", "--quote", "it's \"q\"", "--dollar", "$HOME", "--tick", "`date`"]
    for line, label in ((1, "a b"), (2, "c'd")):
        ind = f"A0_L{line}" if array else f"L{line}"
        outer = ["--tag", "$x y"] if array else []
        assert argv[ind] == fixed + outer + ["--label", label]
        assert logged[ind] == {**AWKWARD, **({"tag": "$x y"} if array else {}), "label": label}

    slurmdunk("collect", config, cwd=tmp_path, check=True)
    con = sqlite3.connect(root / "q" / "results.sqlite")
    rows = dict(con.execute("select indicator, dollar || '|' || label from runs"))
    first = "A0_L1" if array else "L1"
    assert rows[first] == "$HOME|a b"

    # The Python pre-filter and the bash runtime agree on the checkpoint hashes
    rerun = log.with_name("rerun.out")
    run_script(script, rerun, tmp_path)
    assert rerun.read_text().count("[CHECKPOINT] Skipping") == 2
    assert generate(config, tmp_path / "gen")[0] is None
