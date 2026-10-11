"""Shared helpers. Nothing here needs SLURM: generation runs with --dryrun
--notimecheck, generated scripts run under plain bash with a stub executable,
and squeue is replaced by a stub where a test needs it."""
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

# Environment a generated script sees when run outside SLURM: array task 0 of
# job 4242 (array mode) or job 4243 (other modes), with two CPUs so the
# concurrent inner loop runs at most two points at once.
SCRIPT_ENV = {
    "SLURM_ARRAY_JOB_ID": "4242",
    "SLURM_ARRAY_TASK_ID": "0",
    "SLURM_JOB_ID": "4243",
    "SLURM_CPUS_PER_TASK": "2",
    "OMP_NUM_THREADS": "1",
}

STUB = r"""#!/bin/bash
# Prints every argument with its position; writes a small JSON wherever an argument ends in .json
n=0
for a in "$@"; do
    n=$((n + 1))
    printf 'ARGV %d [%s]\n' "$n" "$a"
    case "$a" in *.json) mkdir -p "$(dirname "$a")"; printf '{"metrics": {"result": %d}}\n' "$#" > "$a" ;; esac
done
exit 0
"""


def _bash_at_least(major: int, minor: int) -> bool:
    try:
        out = subprocess.run(["bash", "-c", "echo ${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"],
                             capture_output=True, text=True).stdout.split()
        return (int(out[0]), int(out[1])) >= (major, minor)
    except (OSError, ValueError, IndexError):
        return False


# The generated scripts need bash >= 5.1 (wait -n -p); see "Requirements" in README.md
needs_bash = pytest.mark.skipif(not _bash_at_least(5, 1), reason="generated scripts need bash >= 5.1")


@pytest.fixture
def stub(tmp_path) -> Path:
    path = tmp_path / "stub.sh"
    path.write_text(STUB)
    path.chmod(0o755)
    return path


def write_config(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def slurmdunk(*args, cwd, env=None, check=False) -> subprocess.CompletedProcess:
    """Runs the CLI as 'python -m slurmdunk', in the interpreter running the tests."""
    full_env = dict(os.environ, OMP_NUM_THREADS="1", **(env or {}))
    return subprocess.run([sys.executable, "-m", "slurmdunk", *map(str, args)],
                          cwd=cwd, env=full_env, capture_output=True, text=True, check=check)


def generate(config: Path, workdir: Path):
    """Generates a script with --dryrun --notimecheck; returns (script or None, output)."""
    workdir.mkdir(parents=True, exist_ok=True)
    for old in workdir.glob("submit_*.sh"):
        old.unlink()
    out = slurmdunk("generate", config, "--dryrun", "--notimecheck", cwd=workdir)
    scripts = sorted(workdir.glob("submit_*.sh"))
    return (scripts[0] if scripts else None), out.stdout + out.stderr


def run_script(script: Path, log: Path, cwd: Path, **env) -> int:
    """Runs a generated script locally, writing its output where SLURM would."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.run(["bash", str(script)], cwd=cwd, env={**os.environ, **SCRIPT_ENV, **env},
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def parse_log(log: Path):
    """Per run indicator: the stub's argv, and the logged [<ind>_args] name -> value."""
    argv, logged = {}, {}
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"^\[(\w+?)_out\] ARGV \d+ \[(.*)\]$", line)
        if m:
            argv.setdefault(m.group(1), []).append(m.group(2))
        m = re.match(r"^\[(\w+?)_args\] ([^:]+): ?(.*)$", line)
        if m:
            logged.setdefault(m.group(1), {})[m.group(2)] = m.group(3)
    return argv, logged


def experiment(root: Path, name: str, **extra) -> dict:
    return {"result_database_path": str(root), "experiment_name": name, "slrm_output_dir": "SLURM_OUTPUT", **extra}


SLURM = {"partition": "test", "time": "00:02:00"}


def dedent(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")
