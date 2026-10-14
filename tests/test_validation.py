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


@pytest.mark.parametrize("switches, expected", [
    ({}, (True, True)),
    ({"exec_provenance": False}, (False, True)),
    ({"git_provenance": False}, (True, False)),
    ({"provenance": False}, (False, False)),
    ({"provenance": False, "git_provenance": False}, (False, False)),
])
def test_provenance_switches(switches, expected):
    """provenance: false turns off exec_provenance and git_provenance; each also works alone."""
    config = validate_config({**BASE, "execution": {**BASE["execution"], **switches}}, "cfg.yaml", True)
    assert (config.execution.exec_provenance, config.execution.git_provenance) == expected


@pytest.mark.parametrize("name", ["exec_provenance", "git_provenance"])
def test_provenance_off_with_one_asked_for_is_an_error(name, capsys):
    message = error_of({"execution": {**BASE["execution"], "provenance": False, name: True}}, capsys)
    assert f"'{name}: true' is also set" in message


def test_valid_config_loads():
    config = validate_config({**BASE, "args": {"infile": "a.dat", "n": 3}, "positional_args": ["infile"],
                              "collect": {"patterns": ["time: {t} s"], "json": "{infile}"}}, "cfg.yaml", True)
    assert config.positional_args == ["infile"] and config.positional_placement == "last"
    assert config.argument_sources() == {"infile": ["fixed"], "n": ["fixed"]}


# --------------------------------------------------------------------------- deprecations


@pytest.mark.parametrize("extra, message", [
    ({"loopQ": False}, "'loopQ' is deprecated"),
    ({"execution": {"language": "python3", "executable": "/bin/true"}}, "'execution.language' is deprecated"),
    ({"experiment": {"result_database_path": "/db", "experiment_name": "e", "slrm_output_dir": "o",
                     "tracking_args": {"save_dir": "{save_dir}_{__indicator}"}}}, "use '{run}'"),
])
def test_deprecated_spellings_still_work_and_warn_once(extra, message, capsys, tmp_path):
    config_path = str(tmp_path / "old.yaml")
    config = validate_config({**BASE, **extra}, config_path, True)
    assert config is not None
    assert message in capsys.readouterr().err
    validate_config({**BASE, **extra}, config_path, True)
    assert message not in capsys.readouterr().err  # once per config


def test_current_spellings_do_not_warn(capsys, tmp_path):
    validate_config({**BASE, "execution": {"interpreter": "python3", "executable": "/bin/true"},
                     "experiment": {"result_database_path": "/db", "experiment_name": "e", "slrm_output_dir": "o",
                                    "tracking_args": {"save_dir": "{save_dir}_{run}"}}},
                    str(tmp_path / "new.yaml"), True)
    assert "deprecated" not in capsys.readouterr().err


def test_process_datasets_is_rejected(capsys):
    message = error_of({"process_datasets": True}, capsys)
    assert "not part of SlurmDunk" in message and "Code-Orchestrator" in message
