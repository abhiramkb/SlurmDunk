from typing import Any, Collection, Dict, Tuple

def format_cli_arg(arg: str) -> str:
    """Formats a string as a CLI argument name with appropriate hyphens.
    
    - Single-character names get a single hyphen ('-f')
    - Multi-character names get double hyphens ('--foo')
    - Existing hyphens are preserved ('-f' -> '-f', '--foo' -> '--foo')
    """
    # If the argument already starts with a hyphen, return it as-is
    if arg.startswith('-'):
        return arg
    
    # Single letter gets one hyphen, multi-letter gets two
    return f"-{arg}" if len(arg) == 1 else f"--{arg}"

def format_cli_args(args_dict: Dict[str, Any]) -> str:
    """Formats key-value argument pairs into standard CLI flag strings."""
    args_list = []
    for k, v in args_dict.items():
        flag = format_cli_arg(k)

        # Handle boolean flags / switches
        if isinstance(v, bool):
            if v:  # Only append the flag if True (e.g., '--verbose')
                args_list.append(flag)
        elif v is None:
            continue
        else:
            args_list.append(f'{flag} {v}')

    return " ".join(args_list)

def strip_hyphens(arg: str) -> str:
    """Strips leading hyphens from a CLI argument name string.
    
    Examples:
        '--verbose' -> 'verbose'
        '-v'        -> 'v'
        'foo'       -> 'foo'
        '--foo-bar' -> 'foo-bar'  (preserves internal hyphens)
    """
    return arg.lstrip('-')


def split_positional(args_dict: Dict[str, Any], positional: Collection[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Splits an argument dict into its flag part and its positional part.

    Only the flag part goes through format_cli_args; with no positional
    arguments the flag part is the whole dict, so nothing changes.
    """
    flags = {k: v for k, v in args_dict.items() if k not in positional}
    pos = {k: v for k, v in args_dict.items() if k in positional}
    return flags, pos


class PositionalValueError(ValueError):
    """A value that cannot stand alone in a positional slot."""


def check_positional_value(name: str, value: Any, where: str) -> None:
    """Rejects values that cannot stand alone in a positional slot."""
    if isinstance(value, bool):
        problem = "is true/false; a switch cannot be positional"
    elif value is None:
        problem = "is null; a missing positional value would shift every later one"
    elif str(value) == "":
        problem = "is empty; a missing positional value would shift every later one"
    else:
        return
    raise PositionalValueError(f"positional argument '{name}' {problem} ({where}).")


def bash_dq_escape(value: Any) -> str:
    """Escapes str(value) for use inside a bash double-quoted string."""
    escaped = str(value)
    for ch in ("\\", '"', "$", "`"):
        escaped = escaped.replace(ch, "\\" + ch)
    return escaped


def bash_double_quote(value: Any) -> str:
    """Quotes a value as a bash double-quoted word that expands to exactly str(value)."""
    return f'"{bash_dq_escape(value)}"'


def tracking_template_to_bash(template: str) -> str:
    """Turns a tracking_args template into bash text expanded at run time."""
    return (
        template
        .replace("{result_database_path}", "${RESULT_DATABASE_PATH}")
        .replace("{experiment_name}", "${EXPERIMENT_NAME}")
        .replace("{slrm_output_dir}", "${SLRM_OUTPUT_DIR}")
        .replace("{save_dir}", "${SAVE_DIR}")
        .replace("{__indicator}", "${indicator}")
    )
