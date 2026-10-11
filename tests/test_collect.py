"""'collect' on synthetic SLURM logs: run status, patterns, JSON merging, outputs."""
import csv
import json
import sqlite3

import pytest

from slurmdunk.collect import merge_json, table_cells, write_csv, write_sqlite

from conftest import SLURM, dedent, experiment, slurmdunk, write_config

# Array job 500: task 0 has a completed run (A0_L1), a failed one (A0_L2) and a
# checkpoint skip; task 1 has a run interrupted mid-way. Job 5001 shares a
# prefix with 500, job 600 is a single run, and 'other_500_0.out' belongs to
# another experiment in the same folder.
LOGS = {
    "syn_500_0.out": """
        ======================= SLURM ARRAY RUN LEGEND =======================
        [CHECKPOINT] Skipping A0_L9 - already completed.
        [A0_L1_args] alpha: 0.5
        [A0_L1_args] name: run one
        [A0_L1_args] json: {json}/a0l1.json
        [A0_L1_out] Job started at: Mon Oct  5 12:00:39 AM EEST 2026
        [A0_L1_out] iter 1: value = 1.5
        [A0_L1_err] iter 2: value = 1.25
        [A0_L1_out] iter 3: value = 1.125
        [A0_L1_out] tag: abc123 end
        [A0_L1_out] Job finished at: Mon Oct  5 12:01:00 AM EEST 2026 with Exit Code: 0
        [A0_L1_out] Job duration: 21.004 seconds
        [A0_L2_args] alpha: 0.75
        [A0_L2_args] verbose:
        [A0_L2_args] json: {json}/a0l2.json
        [A0_L2_out] Job started at: Mon
        [A0_L2_err] Traceback: boom
        [A0_L2_out] Job finished at: Mon with Exit Code: 3
        [A0_L2_out] Job duration: 0.500 seconds
        """,
    "syn_500_1.out": """
        [A1_L1_args] alpha: 1.0
        [A1_L1_args] json: {json}/missing.json
        [A1_L1_out] Job started at: Mon
        [A1_L1_out] iter 1: value = 9
        [PARTIAL] ---- incomplete output for A1_L1 (job interrupted) ----
        """,
    "syn_5001.out": """
        [L1_args] alpha: 2
        [L1_out] Job started at: Mon
        [L1_out] Job finished at: Mon with Exit Code: 0
        [L1_out] Job duration: 1.000 seconds
        """,
    "syn_600.out": """
        [SINGLE_args] alpha: 3
        [SINGLE_out] Job started at: Mon
        [SINGLE_out] iter 1: value = 7
        [SINGLE_out] Job finished at: Mon with Exit Code: 0
        [SINGLE_out] Job duration: 2.000 seconds
        """,
    "other_500_0.out": "[A0_L1_args] alpha: 99\n",
}


@pytest.fixture
def experiment_dir(tmp_path):
    root = tmp_path / "db"
    logs = root / "syn" / "SLURM_OUTPUT"
    logs.mkdir(parents=True)
    jdir = tmp_path / "json"
    jdir.mkdir()
    for name, text in LOGS.items():
        (logs / name).write_text(dedent(text).replace("{json}", str(jdir)))
    # alpha equals the argument (merges); 'value' is a pattern name (reserved); 'x'
    # occurs twice; 'status' clashes with a run-info column. a0l2.json is truncated.
    (jdir / "a0l1.json").write_text(json.dumps({
        "parameters": {"alpha": 0.5}, "metrics": {"value": 1.125, "x": 1}, "meta": {"x": 2},
        "status": "ok", "flag": True, "list": [1, 2]}))
    (jdir / "a0l2.json").write_text('{"parameters": {"alpha": 0.7}, "metrics": {"value": 2.0}')
    return root


def make_config(tmp_path, root, collect=None):
    cfg = {
        "execution": {"executable": "/bin/true"},
        "slurm": SLURM,
        "inner_loop": {"type": "explicit", "arg_name": "alpha", "values": [0.5]},
        "experiment": experiment(root, "syn", tracking_args={"json": "x"}),
    }
    if collect is not None:
        cfg["collect"] = collect
    return write_config(tmp_path / "cfg.yaml", cfg)


ALL_FEATURES = {
    "formats": ["sqlite", "csv"],
    "patterns": [
        {"pattern": "value = {value}", "match": "last"},
        {"pattern": "value = {first_value}", "match": "first"},
        {"pattern": "value = {all_values}", "match": "all"},
        {"pattern": "value = {err_value}", "stream": "err"},
        {"regex": r"tag: (?P<tag>\w+)"},
    ],
    "json": {"path": "{json}"},
}


def load(db):
    con = sqlite3.connect(db)
    cur = con.execute("select * from runs")
    cols = [c[0] for c in cur.description]
    return cols, {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}


def test_rows_patterns_and_json(tmp_path, experiment_dir):
    cfg = make_config(tmp_path, experiment_dir, ALL_FEATURES)
    out = slurmdunk("collect", cfg, "--output-dir", tmp_path / "out", cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    cols, rows = load(tmp_path / "out" / "results.sqlite")

    # one row per run that ran: checkpoint skips and the other experiment's log add none
    assert sorted(rows) == ["5001_L1", "500_A0_L1", "500_A0_L2", "500_A1_L1", "600_SINGLE"]
    assert {k: r["status"] for k, r in rows.items()} == {
        "5001_L1": "completed", "500_A0_L1": "completed", "500_A0_L2": "failed",
        "500_A1_L1": "interrupted", "600_SINGLE": "completed"}
    assert rows["500_A0_L2"]["exit_code"] == 3 and rows["500_A1_L1"]["exit_code"] is None
    assert rows["500_A0_L1"]["job_duration"] == 21.004 and rows["500_A1_L1"]["array_task_id"] == 1

    r = rows["500_A0_L1"]
    assert (r["value"], r["first_value"], r["all_values"], r["err_value"], r["tag"]) == \
        (1.125, 1.5, "[1.5, 1.25, 1.125]", 1.25, "abc123")
    assert r["name"] == "run one" and rows["500_A0_L2"]["verbose"] == 1  # a switch is stored as true

    # JSON: merged when equal, dotted path for duplicates and clashes, json. for a run-info name
    assert "parameters.alpha" not in cols and "metrics.value" not in cols
    assert (r["metrics.x"], r["meta.x"], r["json.status"], r["flag"], r["list"]) == (1, 2, "ok", 1, "[1, 2]")
    assert "No JSON found for 3 run(s)" in out.stderr  # missing file, plus two runs without a json arg
    assert "could not be read" in out.stderr

    with open(tmp_path / "out" / "results.csv", newline="") as f:
        body = list(csv.reader(f))
    cell = lambda v: "" if v is None else repr(v) if isinstance(v, float) else str(v)
    assert body[0] == cols
    assert sorted(body[1:]) == sorted([cell(row[c]) for c in cols] for row in rows.values())


def test_job_id_is_exact(tmp_path, experiment_dir):
    cfg = make_config(tmp_path, experiment_dir)
    slurmdunk("collect", cfg, "--job-id", "500", "--output-dir", tmp_path / "o1", cwd=tmp_path, check=True)
    assert sorted(load(tmp_path / "o1" / "results_500.sqlite")[1]) == ["500_A0_L1", "500_A0_L2", "500_A1_L1"]
    slurmdunk("collect", cfg, "--job-id", "500_1", "--output-dir", tmp_path / "o2", cwd=tmp_path, check=True)
    assert sorted(load(tmp_path / "o2" / "results_500_1.sqlite")[1]) == ["500_A1_L1"]


def test_json_null_and_quiet_default(tmp_path, experiment_dir):
    cfg = make_config(tmp_path, experiment_dir, {"json": None})
    out = slurmdunk("collect", cfg, "--output-dir", tmp_path / "o1", cwd=tmp_path, check=True)
    cols = load(tmp_path / "o1" / "results.sqlite")[0]
    assert not {"metrics.x", "meta.x", "json.status", "flag", "list"} & set(cols)
    # Without a collect block the default lookup finds nothing here, and says nothing about it
    cfg = make_config(tmp_path, experiment_dir)
    out = slurmdunk("collect", cfg, "--output-dir", tmp_path / "o2", cwd=tmp_path, check=True)
    assert "No JSON found" not in out.stderr


def test_merge_json_rules():
    row = {"job_id": "1_L1", "alpha": 0.75}
    used = merge_json(row, {"parameters": {"alpha": 0.7}, "metrics": {"value": 3}}, reserved={"value", "alpha"})
    assert used == ["parameters.alpha", "metrics.value"]
    row = {"job_id": "1_L1", "Q": 1.5, "flag": True}
    assert merge_json(row, {"p": {"Q": 1.5, "flag": 1, "nan": float("nan")}}, reserved=set()) == ["p.flag", "nan"]


def test_mixed_type_columns_agree_in_sqlite_and_csv(tmp_path):
    cols = ["job_id", "mixed", "ints_floats", "flag", "lst"]
    rows = [{"job_id": "1_L1", "mixed": 1e-05, "ints_floats": 0, "flag": True, "lst": [1, 2]},
            {"job_id": "1_L2", "mixed": "abc", "ints_floats": 0.5, "flag": None, "lst": None}]
    cells, types = table_cells(cols, rows)
    assert types == ["TEXT", "TEXT", "REAL", "INTEGER", "TEXT"]
    write_sqlite(tmp_path / "t.sqlite", cols, cells, types)
    write_csv(tmp_path / "t.csv", cols, cells, types)
    stored = sqlite3.connect(tmp_path / "t.sqlite").execute("select * from runs").fetchall()
    assert stored == [("1_L1", "1e-05", 0.0, 1, "[1, 2]"), ("1_L2", "abc", 0.5, None, None)]
    with open(tmp_path / "t.csv", newline="") as f:
        assert list(csv.reader(f))[1:] == [["1_L1", "1e-05", "0.0", "1", "[1, 2]"], ["1_L2", "abc", "0.5", "", ""]]
