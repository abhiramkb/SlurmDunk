"""The shipped examples generate, run and collect. They use paths relative to the
repository, so this also covers relative executables and input files."""
import csv
import re
from pathlib import Path

import pytest
import yaml

from conftest import generate, needs_bash, run_script, slurmdunk

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = needs_bash


@pytest.mark.parametrize("name", ["sweep", "positional"])
def test_example_runs(tmp_path, name):
    # Relative paths in the example resolve as they would from the repository root
    (tmp_path / "examples").symlink_to(EXAMPLES)
    cfg = yaml.safe_load((EXAMPLES / f"{name}.yaml").read_text())
    cfg["experiment"]["result_database_path"] = str(tmp_path / "db")
    config = tmp_path / f"{name}.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    script, out = generate(config, tmp_path)
    assert script, out
    logs = tmp_path / "db" / name / "SLURM_OUTPUT"
    for task in ("0", "1"):
        assert run_script(script, logs / f"{name}_4242_{task}.out", tmp_path, SLURM_ARRAY_TASK_ID=task) == 0

    assert slurmdunk("collect", config, "--format", "csv", cwd=tmp_path).returncode == 0
    with open(tmp_path / "db" / name / "results.csv", newline="") as f:
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
