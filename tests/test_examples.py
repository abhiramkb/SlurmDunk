"""The shipped examples generate, run and collect. Their paths are relative to their
own folder, so this also covers config-relative executables, inputs and results."""
import csv
import re
import shutil
from pathlib import Path

import pytest

from conftest import generate, needs_bash, run_script, slurmdunk

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = needs_bash


@pytest.mark.parametrize("name", ["sweep", "positional"])
def test_example_runs(tmp_path, name):
    # The examples as shipped, copied so their results (relative to their folder) stay in tmp_path,
    # and generated from an unrelated directory: relative paths are taken from the config's folder
    examples = tmp_path / "examples"
    shutil.copytree(EXAMPLES, examples, ignore=shutil.ignore_patterns("results", "__pycache__"))
    config = examples / f"{name}.yaml"
    elsewhere = tmp_path / "elsewhere"

    script, out = generate(config, elsewhere)
    assert script, out
    assert f"{examples.resolve()}/print_args.py" in script.read_text()
    logs = examples / "results" / name / "SLURM_OUTPUT"
    for task in ("0", "1"):
        assert run_script(script, logs / f"{name}_4242_{task}.out", elsewhere, SLURM_ARRAY_TASK_ID=task) == 0

    assert slurmdunk("collect", config, "--format", "csv", cwd=elsewhere).returncode == 0
    with open(examples / "results" / name / "results.csv", newline="") as f:
        rows = {r["job_id"]: r for r in csv.DictReader(f)}
    assert len(rows) == 6 and {r["status"] for r in rows.values()} == {"completed"}
    assert rows["4242_A1_L2"]["scale"] == "2.0" and rows["4242_A1_L2"]["x"] == "0.001"

    log = (logs / f"{name}_4242_1.out").read_text()
    if name == "sweep":
        assert {r["mode_echo"] for r in rows.values()} == {"demo"}  # the example's collect pattern
    else:
        # print_args.py reports values it received without a flag as 'positional_<n>'
        assert re.search(r"^\[A1_L2_out\] positional_\d+: 2\.0$", log, re.M)
        assert re.search(r"^\[A1_L2_out\] positional_\d+: 0\.001$", log, re.M)
