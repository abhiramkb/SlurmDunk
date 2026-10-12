"""Bash completion (completions/slurmdunk.bash): its word lists match the CLI, and Tab
suggests the right things."""
import argparse
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from slurmdunk.cli import build_arg_parser

SCRIPT = Path(__file__).resolve().parents[1] / "completions" / "slurmdunk.bash"


def word_list(name):
    match = re.search(rf'^{name}="([^"]*)"$', SCRIPT.read_text(), re.M)
    assert match, f"{name} not found in {SCRIPT.name}"
    return set(match.group(1).split())


def subparsers():
    parser = build_arg_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return parser, action.choices


def option_strings(parser):
    return {s for action in parser._actions for s in action.option_strings}


def test_word_lists_match_the_cli():
    parser, commands = subparsers()
    assert word_list("_slurmdunk_commands") == set(commands)
    assert word_list("_slurmdunk_global_opts") == option_strings(parser)
    for name, subparser in commands.items():
        assert word_list(f"_slurmdunk_{name}_opts") == option_strings(subparser), name
    fmt = next(a for a in commands["collect"]._actions if "--format" in a.option_strings)
    assert word_list("_slurmdunk_formats") == set(fmt.choices)


def test_every_option_taking_a_value_is_handled():
    # Otherwise Tab after e.g. '--job-id' would offer config files instead of nothing
    text = SCRIPT.read_text()
    _, commands = subparsers()
    for subparser in commands.values():
        for action in subparser._actions:
            if action.option_strings and action.nargs != 0 and not isinstance(action, argparse._HelpAction):
                for option in action.option_strings:
                    assert re.search(rf"^\s+{re.escape(option)}\)", text, re.M), option


def complete(words, cwd):
    """What Tab suggests for the command line `words`, whose last word is being completed."""
    line = " ".join(shlex.quote(w) for w in words)
    script = (f"source {shlex.quote(str(SCRIPT))}; COMP_WORDS=({line}); COMP_CWORD={len(words) - 1}; "
              '_slurmdunk; printf "%s\\n" "${COMPREPLY[@]}"')
    out = subprocess.run(["bash", "-c", script], cwd=cwd, capture_output=True, text=True, check=True)
    return sorted(w for w in out.stdout.splitlines() if w)


@pytest.fixture
def folder(tmp_path):
    for name in ("a.yaml", "b.json", "c.yml", "notes.txt"):
        (tmp_path / name).write_text("")
    (tmp_path / "sub").mkdir()
    return tmp_path


@pytest.mark.parametrize("words, expected", [
    (["slurmdunk", ""], ["a.yaml", "b.json", "c.yml", "collect", "generate", "sub"]),  # no notes.txt
    (["slurmdunk", "co"], ["collect"]),
    (["sd", "co"], ["collect"]),                              # the short name
    (["slurmdunk", "--"], ["--help", "--version"]),
    (["slurmdunk", "generate", "--no"], ["--noargcheck", "--nodupcheck", "--notimecheck"]),
    (["slurmdunk", "generate", ""], ["a.yaml", "b.json", "c.yml", "sub"]),
    (["slurmdunk", "a.yaml", "--su"], ["--submit"]),          # a bare config path means generate
    (["slurmdunk", "collect", "a.yaml", "--f"], ["--format"]),
    (["slurmdunk", "collect", "--format", ""], ["both", "csv", "sqlite"]),
    (["slurmdunk", "collect", "--output-dir", ""], ["sub"]),
    (["slurmdunk", "collect", "--job-id", ""], []),
])
def test_suggestions(folder, words, expected):
    assert complete(words, folder) == expected
