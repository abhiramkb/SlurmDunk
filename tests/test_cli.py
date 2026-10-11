"""The command line: subcommands, the bare-config shortcut, version and help."""
import subprocess
import sys

from slurmdunk import __version__

from conftest import SLURM, slurmdunk, write_config


def test_help_lists_subcommands(tmp_path):
    out = slurmdunk("--help", cwd=tmp_path)
    assert out.returncode == 0
    assert "{generate,collect,process}" in out.stdout and "usage: slurmdunk" in out.stdout


def test_version(tmp_path):
    assert slurmdunk("--version", cwd=tmp_path).stdout.strip() == f"slurmdunk {__version__}"


def test_bare_config_path_means_generate(tmp_path, stub):
    cfg = write_config(tmp_path / "c.yaml", {"loopQ": False, "execution": {"executable": str(stub)}, "slurm": SLURM})
    out = slurmdunk(cfg, "--dryrun", "--notimecheck", cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert list(tmp_path.glob("submit_single_*.sh"))


def test_collectold_is_gone(tmp_path):
    # Not a subcommand any more, so it is read as a config path for 'generate'
    out = slurmdunk("collectold", "x.yaml", cwd=tmp_path)
    assert out.returncode == 1 and "Configuration file 'collectold' not found" in out.stderr


def test_python_dash_m(tmp_path):
    out = subprocess.run([sys.executable, "-m", "slurmdunk", "collect", "--help"], cwd=tmp_path,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--format" in out.stdout


def test_run_and_indicator_placeholders_give_the_same_script(tmp_path, stub):
    import re
    scripts = []
    for name, template in (("new", "{save_dir}_{run}"), ("old", "{save_dir}_{__indicator}")):
        cfg = {"execution": {"executable": str(stub)}, "slurm": SLURM,
               "experiment": {"result_database_path": str(tmp_path / "db"), "experiment_name": "e",
                              "slrm_output_dir": "o", "tracking_args": {"save_dir": template}},
               "inner_loop": {"type": "explicit", "arg_name": "x", "values": [1, 2]}}
        config = write_config(tmp_path / f"{name}.yaml", cfg)
        (tmp_path / name).mkdir()
        out = slurmdunk(config, "--dryrun", "--notimecheck", cwd=tmp_path / name)
        assert out.returncode == 0, out.stderr
        script = next((tmp_path / name).glob("submit_*.sh")).read_text()
        scripts.append(re.sub(r"(?m)^(# Generation timestamp:.*|echo \"\[GENERATION\].*|#SBATCH --comment=.*)$", "", script))
    assert scripts[0] == scripts[1]
