"""Config errors are caught when the config is loaded, with a message naming the problem."""
import pytest

from slurmdunk.config_schema import validate_config

BASE = {"execution": {"executable": "/bin/true"}, "slurm": {"partition": "test", "time": "00:02:00"}}


def error_of(extra, capsys):
    with pytest.raises(SystemExit) as exc:
        validate_config({**BASE, **extra}, "cfg.yaml", True)
    assert exc.value.code == 1
    return " ".join(capsys.readouterr().err.split())


@pytest.mark.parametrize("extra, message", [
    ({"args": {"a": 1}, "positional_args": ["b"]}, "'b' is not defined"),
    ({"args": {"a": 1}, "positional_args": ["a", "a"]}, "lists 'a' more than once"),
    ({"args": {"x": 1}, "inner_loop": {"type": "explicit", "arg_name": "x", "values": [1]},
      "positional_args": ["x"]}, "'x' is defined more than once (fixed, inner)"),
    ({"args": {"--a": 1}, "positional_args": ["--a"]}, "has leading dashes"),
    ({"args": {"--a": 1}, "positional_args": ["a"]}, "It is defined as '--a'; drop the dashes there"),
    ({"args": {"v": True}, "positional_args": ["v"]}, "a switch cannot be positional"),
    ({"args": {"v": ""}, "positional_args": ["v"]}, "is empty"),
    ({"args": {"a": 1}, "positional_args": ["a"], "positional_placement": "end"}, "'last' or 'first'"),
])
def test_positional_errors(extra, message, capsys):
    assert message in error_of(extra, capsys)


@pytest.mark.parametrize("collect, message", [
    ({"patterns": ["no placeholder"]}, "has no placeholder"),
    ({"patterns": ["oops {v:float"]}, "unbalanced '{'"),
    ({"patterns": ["a {v}", "b {v}"]}, "placeholder 'v' appears in both"),
    ({"patterns": ["state {status:str}"]}, "reuses a built-in column"),
    ({"patterns": [{"pattern": "a {v}", "regex": "(?P<w>x)"}]}, "exactly one of 'pattern' or 'regex'"),
    ({"patterns": [{"regex": r"x(\d+)"}]}, "has no named group"),
    ({"patterns": [{"patern": "a {v}"}]}, "Extra inputs are not permitted"),
    ({"formats": []}, "at least 1 item"),
])
def test_collect_errors(collect, message, capsys):
    assert message in error_of({"collect": collect}, capsys)


def test_valid_config_loads():
    config = validate_config({**BASE, "args": {"infile": "a.dat", "n": 3}, "positional_args": ["infile"],
                              "collect": {"patterns": ["time: {t} s"], "json": "{infile}"}}, "cfg.yaml", True)
    assert config.positional_args == ["infile"] and config.positional_placement == "last"
    assert config.argument_sources() == {"infile": ["fixed"], "n": ["fixed"]}
