"""Generic result collection, behind the 'collect' subcommand.

Every generated script labels each line of a run's output with the run's
indicator ([A3_L2_args], [A3_L2_out], [A3_L2_err]; see print_args and
run_inner_task), so a SLURM log can be split into runs without knowing anything
about the executable. One row is produced per run: its run info and arguments
always, plus values matched by the config's template patterns and the leaves
of a per-run JSON file.
"""
import csv
import functools
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Pattern, Set, Tuple

# Columns every row carries; placeholder names may not reuse them.
RUN_INFO_COLUMNS = (
    "job_id", "indicator", "array_task_id", "status", "exit_code", "job_duration", "log_file",
)

# Used when the config does not say where a run's JSON is: the per-run directory
# made by the conventional '{save_dir}_{run}' tracking arg.
DEFAULT_JSON_PATH = "{exp_dir}/{job_id}_{run}/result*.json"

FORMAT_CHOICES = {"sqlite": ["sqlite"], "csv": ["csv"], "both": ["sqlite", "csv"]}

# =========================================================================
# Template patterns
# =========================================================================

# Fortran-style D exponents and nan/inf are accepted, as numerical codes print
# them. A trailing '.' is not part of a number, so a sentence-ending period
# does not turn an integer into a float.
_NUMBER_RE = r"[-+]?(?:(?:\d+(?:\.\d+)?|\.\d+)(?:[eEdD][-+]?\d+)?|(?i:nan|inf(?:inity)?)\b)"
_INT_RE = r"[-+]?\d+"


def _to_float(text: str) -> float:
    return float(text.replace("d", "e").replace("D", "E"))


def _to_number(text: str):
    """Keeps the int/float distinction of how the number was written."""
    return int(text) if re.fullmatch(_INT_RE, text) else _to_float(text)


def convert_auto(text: str):
    """Number if the whole text is one, otherwise the text itself.

    Used for argument values and raw-regex captures, which carry no type.
    """
    stripped = text.strip()
    return _to_number(stripped) if re.fullmatch(_NUMBER_RE, stripped) else stripped


# type name -> (regex, converter, is_number). None is the untyped '{x}'.
_PLACEHOLDER_TYPES: Dict[Optional[str], Tuple[str, Callable[[str], Any], bool]] = {
    None: (_NUMBER_RE, _to_number, True),
    "float": (_NUMBER_RE, _to_float, True),
    "int": (_INT_RE, int, True),
    "str": (r"\S+", str, False),
    "text": (r".+", str.strip, False),
}

# '{{' and '}}' are literal braces; '{...}' is a placeholder; any other brace is unbalanced.
_TEMPLATE_TOKEN_RE = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|[{}]")
_PLACEHOLDER_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@functools.lru_cache(maxsize=None)
def compile_template(template: str) -> Tuple[Pattern, Dict[str, Callable[[str], Any]]]:
    """Turns a template like 'Final results: {result} +/- {error}' into a regex.

    Returns the compiled regex and a converter per placeholder name. Raises
    ValueError, naming the template, for anything malformed, so errors surface
    when the config is loaded rather than mid-collection.
    """
    tokens: List[Tuple[str, ...]] = []  # ("lit", text) | ("ph", name, type)
    literal: List[str] = []

    def flush_literal() -> None:
        # Empty text is dropped, so a placeholder at either end really is first/last
        joined = "".join(literal)
        literal.clear()
        if joined:
            tokens.append(("lit", joined))

    text = template.strip()
    pos = 0
    for m in _TEMPLATE_TOKEN_RE.finditer(text):
        literal.append(text[pos:m.start()])
        pos = m.end()
        token = m.group(0)
        if token == "{{":
            literal.append("{")
        elif token == "}}":
            literal.append("}")
        elif m.group(1) is not None:
            flush_literal()
            name, sep, type_name = m.group(1).partition(":")
            name, type_name = name.strip(), (type_name.strip() if sep else None)
            if not _PLACEHOLDER_NAME_RE.match(name):
                raise ValueError(
                    f"placeholder '{{{m.group(1)}}}' in pattern {template!r} needs a name made of "
                    "letters, digits and underscores (it becomes the column name)"
                )
            if type_name not in _PLACEHOLDER_TYPES:
                raise ValueError(
                    f"unknown type '{type_name}' in placeholder '{{{m.group(1)}}}' of pattern "
                    f"{template!r}; use float, int, str or text, or no type for a number"
                )
            if any(t[0] == "ph" and t[1] == name for t in tokens):
                raise ValueError(f"placeholder '{name}' appears twice in pattern {template!r}")
            tokens.append(("ph", name, type_name))
        else:
            raise ValueError(
                f"unbalanced '{token}' in pattern {template!r}; write '{token}{token}' for a literal brace"
            )
    literal.append(text[pos:])
    flush_literal()

    converters = {t[1]: _PLACEHOLDER_TYPES[t[2]][1] for t in tokens if t[0] == "ph"}
    if not converters:
        raise ValueError(f"pattern {template!r} has no placeholder such as '{{value}}'")

    parts: List[str] = []
    for i, token in enumerate(tokens):
        if token[0] == "lit":
            # Any run of whitespace matches any (non-empty) run, so aligned output still matches
            for chunk in re.split(r"(\s+)", token[1]):
                if chunk:
                    parts.append(r"\s+" if chunk.isspace() else re.escape(chunk))
            continue

        _, name, type_name = token
        body, _, is_number = _PLACEHOLDER_TYPES[type_name]
        first, last = i == 0, i == len(tokens) - 1
        if is_number:
            # Guards only at the template's edges; elsewhere the neighbouring literal
            # delimits the number. They stop e.g. '{n:int}' matching the 5 of '10.5'.
            before = r"(?<![\w.])" if first else ""
            after = r"(?!\.?\d)" if last else ""
            parts.append(f"{before}(?P<{name}>{body}){after}")
        else:
            # Text stops at the next literal, or runs to the end of the line when last
            parts.append(f"(?P<{name}>{body if last else body + '?'})")

    return re.compile("".join(parts)), converters


@functools.lru_cache(maxsize=None)
def compile_regex(regex: str) -> Tuple[Pattern, Dict[str, Callable[[str], Any]]]:
    """The 'regex:' escape hatch: every named group becomes a column."""
    try:
        compiled = re.compile(regex)
    except re.error as e:
        raise ValueError(f"invalid regex {regex!r}: {e}") from e
    if not compiled.groupindex:
        raise ValueError(f"regex {regex!r} has no named group such as '(?P<value>...)'")
    return compiled, {name: convert_auto for name in compiled.groupindex}


def pattern_matcher(spec) -> Tuple[Pattern, Dict[str, Callable[[str], Any]]]:
    return compile_template(spec.pattern) if spec.pattern is not None else compile_regex(spec.regex)


def apply_patterns(lines: List[Tuple[str, str]], patterns: List[Any]) -> Dict[str, Any]:
    """Matches each pattern against one run's (stream, text) lines."""
    values: Dict[str, Any] = {}
    for spec in patterns:
        regex, converters = pattern_matcher(spec)
        hits = []
        for stream, text in lines:
            if spec.stream != "any" and stream != spec.stream:
                continue
            m = regex.search(text)
            if m:
                hits.append({
                    name: (None if m.group(name) is None else conv(m.group(name)))
                    for name, conv in converters.items()
                })
                if spec.match == "first":
                    break
        if not hits:
            continue
        if spec.match == "all":
            for name in converters:
                values[name] = [hit[name] for hit in hits]
        else:
            values.update(hits[0] if spec.match == "first" else hits[-1])
    return values


# =========================================================================
# SLURM log parsing
# =========================================================================

_LOG_LINE_RE = re.compile(r"^\[(SINGLE|(?:A\d+_)?L\d+)_(args|out|err)\] ?(.*)$")
_FINISHED_RE = re.compile(r"^Job finished at: .* with Exit Code: (-?\d+)\s*$")
_DURATION_RE = re.compile(r"^Job duration: ([\d.]+) seconds\s*$")


def find_log_files(
    slurm_dir: Path, exp_name: str, job_id: Optional[str] = None
) -> List[Tuple[Path, str, Optional[int]]]:
    """Returns (path, job id, array task id) for every log of the experiment.

    Relies on the names setup_experiment_directories gives the logs:
    <exp>_<jobid>.out, or <exp>_<arrayjobid>_<task>.out for arrays. job_id is
    matched exactly, either as a job ID or as '<arrayjobid>_<task>'.
    """
    name_re = re.compile(rf"^{re.escape(exp_name)}_(\d+)(?:_(\d+))?\.out$")
    found = []
    for entry in os.scandir(slurm_dir):
        m = name_re.match(entry.name)
        if not m or not entry.is_file():
            continue
        if job_id is not None and str(job_id) not in (m.group(1), f"{m.group(1)}_{m.group(2)}"):
            continue
        found.append((Path(entry.path), m.group(1), int(m.group(2)) if m.group(2) else None))
    found.sort(key=lambda f: (int(f[1]), -1 if f[2] is None else f[2]))
    return found


def _new_run(indicator: str) -> Dict[str, Any]:
    return {
        "indicator": indicator, "args": {}, "lines": [],
        "started": False, "finished": False, "exit_code": None, "duration": None,
    }


def parse_runs(log_path: Path) -> List[Dict[str, Any]]:
    """Splits one SLURM log into runs, in order of first appearance.

    The script's own timing lines become run info and are kept out of the
    lines the patterns see. Lines without a run label (the legend, checkpoint
    skips) belong to no run, so a skipped run produces no row here; its row
    comes from the log of the job that actually ran it.
    """
    runs: Dict[str, Dict[str, Any]] = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = _LOG_LINE_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            indicator, stream, text = m.groups()
            run = runs.get(indicator)

            if stream == "args":
                if run is None or run["started"]:
                    # Arguments printed again after a start mean the run was restarted
                    # into the same log (e.g. a requeue); its latest attempt counts.
                    run = runs[indicator] = _new_run(indicator)
                name, sep, value = text.partition(":")
                if sep:
                    run["args"][name.strip()] = value.strip()
                continue

            if run is None:
                run = runs[indicator] = _new_run(indicator)
            if stream == "out":
                if text.startswith("Job started at: "):
                    run["started"] = True
                    continue
                finished = _FINISHED_RE.match(text)
                if finished:
                    run["finished"] = True
                    run["exit_code"] = int(finished.group(1))
                    continue
                duration = _DURATION_RE.match(text)
                if duration:
                    run["duration"] = float(duration.group(1))
                    continue
            run["lines"].append((stream, text))
    return list(runs.values())


def run_status(run: Dict[str, Any]) -> str:
    if not run["finished"]:
        return "interrupted"  # walltime, cancellation or node failure mid-run
    return "completed" if run["exit_code"] == 0 else "failed"


# =========================================================================
# JSON
# =========================================================================

class _StrictFormatDict(dict):
    def __missing__(self, key):
        raise KeyError(key)


def load_json(path_template: str, placeholders: Dict[str, str]) -> Tuple[Optional[dict], Optional[str]]:
    """Returns (document, None), or (None, reason) with reason 'missing' or 'invalid'."""
    try:
        path = path_template.format_map(_StrictFormatDict(placeholders))
    except (KeyError, IndexError, ValueError):
        return None, "missing"  # the run did not print an argument the path refers to
    if any(c in path for c in "*?["):
        matches = sorted(glob(path))
        if not matches:
            return None, "missing"
        path = matches[0]
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError):
        return None, "invalid"
    return (doc, None) if isinstance(doc, dict) else (None, "invalid")


def _iter_leaves(doc: dict, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], Any]]:
    for key, value in doc.items():
        if isinstance(value, dict) and value:
            yield from _iter_leaves(value, path + (str(key),))
        else:
            yield path + (str(key),), value


def _value_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    return "number" if isinstance(value, (int, float)) else type(value).__name__


def _same_value(a: Any, b: Any) -> bool:
    if _value_kind(a) != _value_kind(b):
        return False
    if _value_kind(a) == "number":
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def merge_json(row: Dict[str, Any], doc: dict, reserved: Set[str]) -> List[str]:
    """Adds a JSON document's leaves to a row; returns the column names it used.

    A leaf goes under its own name. It is merged into an existing column holding
    the same value (e.g. parameters.Q into the Q argument), and keeps its dotted
    path when the name is taken by a different value or occurs twice in the
    document. reserved names (patterns, arguments) count as taken even in rows
    where they are empty, so such a column never mixes in JSON values. A dotted
    path that is itself taken (a top-level key) gets 'json.'.
    """
    leaves = list(_iter_leaves(doc))
    counts = Counter(path[-1] for path, _ in leaves)
    used = []
    for path, value in leaves:
        name = path[-1]
        if counts[name] > 1:
            name = ".".join(path)
        elif name in row or name in reserved:
            if name in row and _same_value(row[name], value):
                continue
            name = ".".join(path)
        if name in row:
            name = "json." + ".".join(path)
        row[name] = value
        used.append(name)
    return used


# =========================================================================
# Output
# =========================================================================

def _cell(value: Any) -> Any:
    """Converts a value to what both SQLite and CSV store."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    if isinstance(value, int) and not -2**63 <= value < 2**63:
        return str(value)  # beyond SQLite's INTEGER range
    return value


def _sql_type(values: List[Any]) -> str:
    kinds = {
        "TEXT" if isinstance(v, str) else "REAL" if isinstance(v, float) else "INTEGER"
        for v in values if v is not None
    }
    if not kinds or "TEXT" in kinds:
        return "TEXT"
    return "REAL" if "REAL" in kinds else "INTEGER"


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_cells(columns: List[str], rows: List[Dict[str, Any]]) -> Tuple[List[List[Any]], List[str]]:
    """The cell values and SQLite type of every column, shared by both writers.

    Each column takes one type: ints in a column that also holds floats become
    floats, and numbers in a column that also holds text become text. SQLite
    would otherwise coerce them by column affinity (0 -> 0.0, 1e-05 -> '1.0e-05')
    while CSV kept them as written, and the two outputs would disagree.
    """
    cells = [[_cell(row.get(c)) for c in columns] for row in rows]
    types = [_sql_type([r[i] for r in cells]) for i in range(len(columns))]
    types[0] = "TEXT"  # job_id, the primary key
    convert = {"REAL": float, "TEXT": str}
    for i, col_type in enumerate(types):
        if col_type in convert:
            for r in cells:
                if r[i] is not None:
                    r[i] = convert[col_type](r[i])
    return cells, types


def write_sqlite(path: Path, columns: List[str], cells: List[List[Any]], types: List[str]) -> None:
    """Rebuilds the 'runs' table from scratch, replacing the file atomically."""
    col_defs = [f"{_quote(columns[0])} TEXT PRIMARY KEY"] + [
        f"{_quote(c)} {t}" for c, t in zip(columns[1:], types[1:])
    ]

    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        conn.execute(f"CREATE TABLE runs ({', '.join(col_defs)})")
        conn.executemany(
            f"INSERT INTO runs ({', '.join(map(_quote, columns))}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            cells,
        )
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, path)


def write_csv(path: Path, columns: List[str], cells: List[List[Any]], types: List[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in cells:
            writer.writerow(["" if v is None else v for v in row])
    os.replace(tmp, path)


# =========================================================================
# Entry point
# =========================================================================

def collect_results(
    config: Any,
    arg_names: Set[str],
    job_id: Optional[str] = None,
    formats: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Collects one row per run from an experiment's SLURM logs.

    config is the validated AppConfig; arg_names are the argument names the
    config passes to the executable, which placeholder names may not reuse.
    formats defaults to collect.formats from the config.
    """
    exp = config.experiment
    if exp is None:
        print(
            "[ERROR] 'collect' needs an 'experiment' block: the logs are read from "
            "<result_database_path>/<experiment_name>/<slrm_output_dir>.",
            file=sys.stderr,
        )
        sys.exit(1)

    collect_cfg = config.collect
    patterns = collect_cfg.patterns if collect_cfg else []
    json_spec = collect_cfg.json_file if collect_cfg else None
    if collect_cfg is None:
        json_path, json_explicit = DEFAULT_JSON_PATH, False
    elif json_spec is None:
        json_path, json_explicit = None, False  # 'json: null' turns JSON off
    else:
        json_path, json_explicit = json_spec.path, "path" in json_spec.model_fields_set
    formats = formats or (collect_cfg.formats if collect_cfg else ["sqlite"])

    for spec in patterns:
        clashes = sorted(set(pattern_matcher(spec)[1]) & arg_names)
        if clashes:
            print(
                f"[ERROR] Placeholder(s) {', '.join(clashes)} in collect pattern "
                f"{(spec.pattern or spec.regex)!r} reuse argument name(s); rename the placeholder(s).",
                file=sys.stderr,
            )
            sys.exit(1)

    exp_dir = Path(exp.result_database_path) / exp.experiment_name
    slurm_dir = exp_dir / exp.slrm_output_dir
    if not slurm_dir.is_dir():
        print(f"[ERROR] SLURM output directory not found: {slurm_dir}", file=sys.stderr)
        sys.exit(1)

    logs = find_log_files(slurm_dir, exp.experiment_name, job_id)
    if not logs:
        which = f" for job {job_id}" if job_id is not None else ""
        print(f"[WARNING] No SLURM logs{which} found in {slurm_dir}; nothing written.", file=sys.stderr)
        return

    pattern_names = [name for spec in patterns for name in pattern_matcher(spec)[1]]
    reserved = set(pattern_names) | arg_names
    arg_columns: Dict[str, None] = {}
    json_columns: Dict[str, None] = {}
    pattern_hits = Counter()
    json_problems = Counter()
    rows: Dict[str, Dict[str, Any]] = {}
    duplicates = 0

    for log_path, base_job_id, task_id in logs:
        for run in parse_runs(log_path):
            row: Dict[str, Any] = {
                "job_id": f"{base_job_id}_{run['indicator']}",
                "indicator": run["indicator"],
                "array_task_id": task_id,
                "status": run_status(run),
                "exit_code": run["exit_code"],
                "job_duration": run["duration"],
                "log_file": log_path.name,
            }

            for name, raw in run["args"].items():
                column = f"arg.{name}" if name in RUN_INFO_COLUMNS else name
                # A switch is printed with an empty value
                row[column] = True if raw == "" else convert_auto(raw)
                arg_columns.setdefault(column)

            for spec in patterns:
                matched = apply_patterns(run["lines"], [spec])
                if matched:
                    pattern_hits[spec.pattern or spec.regex] += 1
                    row.update(matched)

            if json_path is not None:
                # Built-ins take precedence over arguments of the same name
                placeholders = dict(run["args"])
                placeholders.update(job_id=base_job_id, run=run["indicator"], indicator=run["indicator"],
                                    exp_dir=str(exp_dir))
                doc, problem = load_json(json_path, placeholders)
                if doc is not None:
                    for column in merge_json(row, doc, reserved):
                        json_columns.setdefault(column)
                else:
                    json_problems[problem] += 1

            if row["job_id"] in rows:
                duplicates += 1
            rows[row["job_id"]] = row

    columns = (
        list(RUN_INFO_COLUMNS) + list(arg_columns)
        + [n for n in pattern_names if n not in arg_columns]
        + [c for c in json_columns if c not in arg_columns and c not in pattern_names]
    )
    row_list = list(rows.values())

    statuses = Counter(row["status"] for row in row_list)
    print(
        f"[INFO] {len(row_list)} run(s) from {len(logs)} log(s): "
        + (", ".join(f"{statuses[s]} {s}" for s in ("completed", "failed", "interrupted") if statuses[s])
           or "nothing ran")
    )
    for spec in patterns:
        label = spec.pattern or spec.regex
        print(f"[INFO] Pattern {label!r} matched in {pattern_hits[label]}/{len(row_list)} run(s).")
    if duplicates:
        print(f"[WARNING] {duplicates} run(s) appeared in more than one log; the last occurrence was kept.",
              file=sys.stderr)
    if json_problems["missing"] and json_explicit:
        print(f"[WARNING] No JSON found for {json_problems['missing']} run(s) (json.path: {json_path}).",
              file=sys.stderr)
    if json_problems["invalid"]:
        print(f"[WARNING] {json_problems['invalid']} JSON file(s) could not be read as a JSON object.",
              file=sys.stderr)

    out_dir = Path(output_dir) if output_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "results" if job_id is None else f"results_{job_id}"
    cells, types = table_cells(columns, row_list)
    writers = {"sqlite": (write_sqlite, ".sqlite"), "csv": (write_csv, ".csv")}
    for fmt in dict.fromkeys(formats):
        writer, suffix = writers[fmt]
        out_path = out_dir / f"{stem}{suffix}"
        writer(out_path, columns, cells, types)
        print(f"[SUCCESS] Wrote {len(row_list)} row(s) x {len(columns)} column(s) to {out_path}")
