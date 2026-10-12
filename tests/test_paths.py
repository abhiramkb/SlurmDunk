"""Relative paths in a config are taken from the config file's folder, wherever
slurmdunk is run from."""
import shutil
from pathlib import Path

import pytest

from slurmdunk.config_schema import validate_config

from conftest import SLURM, generate, needs_bash, parse_log, run_script, slurmdunk, write_config


@needs_bash
def test_relative_paths_follow_the_config(tmp_path, stub):
    project = tmp_path / "project"
    (project / "bin").mkdir(parents=True)
    shutil.copy(stub, project / "bin" / "prog.sh")
    (project / "points.txt").write_text("# x\n0.1\n0.2\n")
    config = write_config(project / "cfg.yaml", {
        "execution": {"executable": "bin/prog.sh"},
        "slurm": SLURM,
        "experiment": {"result_database_path": "out", "experiment_name": "rel", "slrm_output_dir": "logs",
                       "tracking_args": {"save_dir": "{save_dir}_{run}"}},
        "inner_loop": {"type": "tabular_file", "file_path": "points.txt", "arg_names": ["x"]},
    })
    elsewhere = tmp_path / "elsewhere"
    script, out = generate(config, elsewhere)
    assert script, out
    text = script.read_text()
    base = project.resolve()
    assert f"{base}/bin/prog.sh" in text and f'RESULT_DATABASE_PATH="{base}/out"' in text

    log = base / "out" / "rel" / "logs" / "rel_4243.out"
    assert run_script(script, log, elsewhere) == 0
    argv, _ = parse_log(log)
    assert argv == {"L1": ["-x", "0.1", "--save_dir", f"{base}/out/rel/4243_L1"],
                    "L2": ["-x", "0.2", "--save_dir", f"{base}/out/rel/4243_L2"]}

    # collect, also run from elsewhere, finds the logs under the config's folder
    slurmdunk("collect", config, cwd=elsewhere, check=True)
    assert (base / "out" / "rel" / "results.db").is_file()


def test_bare_command_is_left_for_path_lookup(tmp_path):
    folder = tmp_path / "a"
    folder.mkdir()
    config = str(folder / "cfg.yaml")
    cfg = {"execution": {"executable": "echo"}, "slurm": SLURM}
    assert validate_config(cfg, config, True).execution.executable == "echo"
    # ... unless a file of that name sits next to the config
    (folder / "echo").write_text("")
    assert validate_config(cfg, config, True).execution.executable == str((folder / "echo").resolve())


def test_other_paths_resolve_against_the_config(tmp_path):
    folder = (tmp_path / "cfgdir").resolve()
    folder.mkdir()
    config = str(folder / "cfg.yaml")
    base = {"execution": {"interpreter": "python3", "executable": "prog.py"}, "slurm": SLURM}
    c = validate_config({**base, "collect": {"json": "res/{run}.json"}}, config, True)
    assert c.execution.executable == f"{folder}/prog.py"
    assert c.collect.json_file.path == f"{folder}/res/{{run}}.json"
    assert validate_config({**base, "collect": {"json": "{json}"}}, config, True).collect.json_file.path == "{json}"
    assert validate_config({**base, "experiment": {"result_database_path": "/abs/db", "experiment_name": "e",
                                                   "slrm_output_dir": "o"}}, config, True).experiment.result_database_path == "/abs/db"
    p = validate_config({**base, "outer_loops": [{"type": "tabular_file", "file_path": "t.txt",
                                                  "args": [{"arg_name": "a", "column": 0}]}],
                         "inner_loop": {"type": "explicit", "arg_name": "b", "values": [1]}}, config, True)
    assert p.outer_loops[0].file_path == folder / "t.txt"
