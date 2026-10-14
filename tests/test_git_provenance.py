"""execution.git_provenance: the executable's git state is logged before every run
and becomes columns in 'collect'."""
import json
import shutil
import sqlite3
import subprocess

import pytest

from conftest import SLURM, STUB, experiment, generate, needs_bash, run_script, slurmdunk, write_config

pytestmark = [needs_bash, pytest.mark.skipif(shutil.which("git") is None, reason="needs git")]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repository with a committed executable (the stub) and one other tracked file."""
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    (path / "prog.sh").write_text(STUB)
    (path / "prog.sh").chmod(0o755)
    (path / "notes.txt").write_text("v1\n")
    git(path, "add", "prog.sh", "notes.txt")
    git(path, "commit", "-q", "-m", "initial")
    return path


def run_and_read(tmp_path, executable, points=(1,), **execution):
    """Generates an inner-loop sweep over 'executable', runs it, and returns the logged git lines
    per run plus the config path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "db"
    config = write_config(tmp_path / "cfg.yaml", {
        "execution": {"executable": str(executable), **execution},
        "slurm": SLURM,
        "experiment": experiment(root, "g"),
        "inner_loop": {"type": "explicit", "arg_name": "x", "values": list(points)},
    })
    script, out = generate(config, tmp_path / "gen")
    assert script, out
    log = root / "g" / "SLURM_OUTPUT" / "g_4243.out"
    assert run_script(script, log, tmp_path) == 0
    logged = {}
    for line in log.read_text().splitlines():
        if line.startswith("[L") and "_git] " in line:
            ind, rest = line[1:].split("_git] ", 1)
            name, _, value = rest.partition(": ")
            logged.setdefault(ind, {})[name] = value
    return logged, config, script


def test_clean_repository(tmp_path, repo):
    logged, config, _ = run_and_read(tmp_path, repo / "prog.sh", points=(1, 2))
    head = git(repo, "rev-parse", "HEAD")
    expected = {"repo": str(repo.resolve()), "commit": head, "repo_dirty": "false", "exec_dirty": "false"}
    assert logged == {"L1": expected, "L2": expected}

    slurmdunk("collect", config, cwd=tmp_path, check=True)
    con = sqlite3.connect(tmp_path / "db" / "g" / "results.db")
    cols = [c[1] for c in con.execute("pragma table_info(runs)")]
    assert cols[7:12] == ["executable", "git_repo", "git_commit", "git_repo_dirty", "git_exec_dirty"]
    assert set(con.execute("select executable, git_commit, git_repo_dirty, git_exec_dirty from runs")) == \
        {(str((repo / "prog.sh").resolve()), head, 0, 0)}


@pytest.mark.parametrize("change, repo_dirty, exec_dirty", [
    ("tracked file modified", "true", "false"),
    ("untracked file added", "false", "false"),   # results lying around don't make the code dirty
    ("executable modified", "true", "true"),
    ("executable staged", "true", "true"),
])
def test_dirty_states(tmp_path, repo, change, repo_dirty, exec_dirty):
    if change == "tracked file modified":
        (repo / "notes.txt").write_text("v2\n")
    elif change == "untracked file added":
        (repo / "result.json").write_text("{}")
    else:
        (repo / "prog.sh").write_text(STUB + "# edited\n")
        if change == "executable staged":
            git(repo, "add", "prog.sh")
    logged, _, _ = run_and_read(tmp_path, repo / "prog.sh")
    assert (logged["L1"]["repo_dirty"], logged["L1"]["exec_dirty"]) == (repo_dirty, exec_dirty)


def test_untracked_executable_counts_as_dirty(tmp_path, repo):
    shutil.copy(repo / "prog.sh", repo / "new_prog.sh")
    logged, _, _ = run_and_read(tmp_path, repo / "new_prog.sh")
    assert logged["L1"]["exec_dirty"] == "true" and logged["L1"]["repo_dirty"] == "false"


def test_ignored_executable_counts_as_dirty(tmp_path, repo):
    # e.g. a compiled binary in an ignored build directory: git status doesn't list it at all
    (repo / ".gitignore").write_text("build/\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore build")
    (repo / "build").mkdir()
    shutil.copy(repo / "prog.sh", repo / "build" / "prog")
    logged, _, _ = run_and_read(tmp_path, repo / "build" / "prog")
    assert logged["L1"]["exec_dirty"] == "true" and logged["L1"]["repo_dirty"] == "false"


def test_recorded_before_every_run(tmp_path, repo):
    # A program that edits a tracked file: the first run starts clean, the second sees the change
    (repo / "prog.sh").write_text(STUB.replace("exit 0", f'echo edited >> "{repo}/notes.txt"\nexit 0'))
    git(repo, "commit", "-q", "-am", "edits notes")
    logged, _, _ = run_and_read(tmp_path, repo / "prog.sh", points=(1, 2))
    assert logged["L1"]["repo_dirty"] == "false" and logged["L2"]["repo_dirty"] == "true"


def test_outside_a_repository_and_bare_commands(tmp_path, stub):
    logged, config, _ = run_and_read(tmp_path, stub)
    assert logged == {"L1": {"repo": "none"}}
    slurmdunk("collect", config, cwd=tmp_path, check=True)
    con = sqlite3.connect(tmp_path / "db" / "g" / "results.db")
    assert list(con.execute("select git_repo, git_commit, git_repo_dirty, git_exec_dirty from runs")) == \
        [(None, None, None, None)]
    # A bare command is looked up on PATH when the job runs; 'true' is not in a repository
    logged, _, _ = run_and_read(tmp_path / "bare", "true")
    assert logged == {"L1": {"repo": "none"}}


def test_off_switch(tmp_path, repo):
    logged, _, script = run_and_read(tmp_path, repo / "prog.sh", git_provenance=False)
    assert logged == {} and "record_git" not in script.read_text()


def test_json_git_commit_merges_with_the_column(tmp_path, repo):
    """A program writing its own git_commit into its JSON (as the NLO codes do) gives one column
    when the values agree, and keeps its own under the dotted path when they don't."""
    head = git(repo, "rev-parse", "HEAD")
    root = tmp_path / "db"
    config = write_config(tmp_path / "cfg.yaml", {
        "execution": {"executable": str(repo / "prog.sh")},
        "slurm": SLURM,
        "experiment": experiment(root, "g", tracking_args={"json": "{save_dir}_{run}/result.json"}),
        "inner_loop": {"type": "explicit", "arg_name": "x", "values": [1, 2]},
    })
    script, _ = generate(config, tmp_path / "gen")
    assert run_script(script, root / "g" / "SLURM_OUTPUT" / "g_4243.out", tmp_path) == 0
    for run, commit in (("L1", head), ("L2", "0123abc")):
        (root / "g" / f"4243_{run}" / "result.json").write_text(json.dumps({"provenance": {"git_commit": commit}}))
    slurmdunk("collect", config, cwd=tmp_path, check=True)
    con = sqlite3.connect(root / "g" / "results.db")
    rows = {r[0]: r[1:] for r in con.execute('select indicator, git_commit, "provenance.git_commit" from runs')}
    assert rows == {"L1": (head, None), "L2": (head, "0123abc")}
