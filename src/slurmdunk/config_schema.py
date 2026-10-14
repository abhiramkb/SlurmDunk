import re
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Tuple, Union
from pydantic import AliasChoices, BaseModel, Field, ValidationError, ValidationInfo, computed_field, field_validator, model_validator

# Pattern compilation is shared with the collector, so template errors surface when
# the config is loaded rather than mid-collection
from .collect import BUILTIN_COLUMNS, DEFAULT_JSON_PATH, pattern_matcher
from .utils import check_positional_value, format_cli_arg

def resolve_config_path(path: Union[str, Path], info: ValidationInfo) -> Path:
    """Makes a relative path absolute, taken from the config file's folder.

    validate_config puts that folder in the validation context as 'base_dir'; without
    it (a model validated directly) the current directory is used. Absolute paths are
    returned unchanged, so a config gives the same paths wherever slurmdunk is run.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    base = info.context.get("base_dir") if info.context else None
    return Path(base) / path if base else path.resolve()


def determine_loop_q(cfg: dict) -> bool:
    """Determines if loop mode is active based on config flags and structure."""
    mode_index = 0 # 0 -> no loops, 1 -> inner loop only, 2 -> inner and outer loops

    if "loopQ" in cfg:
        mode_index = 1
        if "outer_loops" in cfg:
            mode_index = 2
        return True, mode_index

    # If loopQ is missing, check if loop options are absent while args is present

    has_inner = "inner_loop" in cfg
    has_outer = "outer_loops" in cfg
    has_args = "args" in cfg
    has_slurm = "slurm" in cfg
    has_execution = "execution" in cfg

    if not has_inner and not has_outer and has_args and has_slurm and has_execution:
        return False, mode_index

    if has_inner:
        mode_index = 1
    if has_outer:
        mode_index = 2

    return True, mode_index


# --- 1. Outer Loop Polymorphic Models ---

class ExplicitLoop(BaseModel):
    type: Literal["explicit"] = "explicit"
    arg_name: str
    values: List[Any]


class RangeLoop(BaseModel):
    type: Literal["range"]
    arg_name: str
    start: float
    stop: float
    step: float = 1.0


class TabularArgSpec(BaseModel):
    arg_name: str
    # 'column' takes the value from a column of the row. 'row_index' takes the row's
    # position among retained rows instead, so comment and blank lines do not advance
    # it: with a 4-line header, the first data row is index 0.
    source: Literal["column", "row_index"] = "column"
    column: Optional[int] = None
    index_offset: int = 0  # Added to the row index, e.g. 1 for one-based numbering
    template: Optional[str] = None
    transform: Optional[str] = None  # e.g., "10^x", "10**x", "log10", "exp"

    @model_validator(mode="after")
    def check_source_fields(self) -> "TabularArgSpec":
        """Keeps the two sources mutually exclusive instead of silently ignoring
        whichever field does not apply to the chosen source."""
        if self.source == "column":
            if self.column is None:
                raise ValueError(
                    f"'{self.arg_name}': 'column' is required unless 'source' is 'row_index'."
                )
            if self.index_offset:
                raise ValueError(
                    f"'{self.arg_name}': 'index_offset' only applies to source 'row_index'."
                )
        elif self.column is not None:
            raise ValueError(
                f"'{self.arg_name}': 'column' cannot be combined with source 'row_index'."
            )
        return self


class TabularOuterLoop(BaseModel):
    type: Literal["tabular_file"]
    file_path: Path
    args: List[TabularArgSpec]
    delimiter: Optional[str] = None
    comment_prefix: Union[str, List[str]] = "#"  # a single prefix, or several, e.g. ["#", "*", "!"]
    skip_blank_lines: bool = True

    @field_validator("file_path")
    @classmethod
    def check_file_exists(cls, v: Path, info: ValidationInfo) -> Path:
        v = resolve_config_path(v, info)
        # Reads 'check_files' from context, defaulting to True if not provided
        check_files = info.context.get("check_files", True) if info.context else True        
        if check_files and not v.is_file():
            raise ValueError(f"Tabular file does not exist: '{v}'")
        return v


# Tagged Union discriminator routes validation based on the 'type' field
OuterLoopBlock = Annotated[
    Union[ExplicitLoop, RangeLoop, TabularOuterLoop],
    Field(discriminator="type"),
]


# --- 2. Sub-Section Models ---

class ExecutionConfig(BaseModel):
    # 'language' is the legacy spelling of 'interpreter'. Both are accepted; an explicit
    # 'interpreter' wins if a config carries both.
    interpreter: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("interpreter", "language")
    )
    flags: Optional[List[str]] = Field(default_factory=list)
    executable: str
    modules: List[str] = Field(default_factory=list)
    env_vars: Dict[str, Any] = Field(default_factory=dict)
    preamble: Optional[str] = None
    # Free-form bash run once every executable invocation in the script has finished.
    # In array mode that is once per array element, mirroring the preamble.
    epilogue: Optional[str] = None
    # Cores used by ONE invocation of the executable. When set, the inner loop runs
    # floor(SLURM_CPUS_PER_TASK / multithreading_level) iterations concurrently.
    # Omit it (the default) to keep the inner loop strictly sequential.
    # ge=1 also guards the runtime division in the generated script.
    multithreading_level: Optional[int] = Field(default=None, ge=1)
    # Optional hard ceiling on concurrency, e.g. to stay within --mem.
    max_parallel_tasks: Optional[int] = Field(default=None, ge=1)
    # What is logged about the executable before every run (see "Provenance" in README.md).
    # provenance: false turns off both of the others.
    provenance: bool = True
    # Its path, as '[<run>_exec] path: ...'
    exec_provenance: bool = True
    # Its git commit, and whether it or its repository has uncommitted changes
    git_provenance: bool = True

    @model_validator(mode="after")
    def apply_provenance_switch(self) -> "ExecutionConfig":
        """provenance: false turns exec_provenance and git_provenance off. Asking for
        either explicitly at the same time is a contradiction, not a precedence rule."""
        if self.provenance:
            return self
        for name in ("exec_provenance", "git_provenance"):
            if name in self.model_fields_set and getattr(self, name):
                raise ValueError(
                    f"'provenance: false' turns off both exec_provenance and git_provenance, but "
                    f"'{name}: true' is also set. Drop one of them: to keep only {name}, set "
                    f"provenance to true and the other one to false."
                )
        self.exec_provenance = self.git_provenance = False
        return self

    @field_validator("executable")
    @classmethod
    def resolve_executable(cls, v: str, info: ValidationInfo) -> str:
        """A path to the program is taken from the config file's folder. A bare name
        without an interpreter (e.g. 'mytool') is a command found on PATH, kept as is,
        unless a file of that name sits next to the config."""
        if info.data.get("interpreter") or "/" in v:
            return str(resolve_config_path(v, info))
        local = resolve_config_path(v, info)
        return str(local) if local.is_file() else v


class SlurmConfig(BaseModel):
    job_name: str = Field(default="slurmdunk", alias="job-name")
    partition: str
    nodes: int = Field(default=1, alias="nodes") # Default number of nodes is 1 if not specified
    time: str
    output: str = Field(default="slurm_%j.log", alias="output") # Output set in setup_experiment_directories function if experiment tracking info is provided.
    max_concurrent_tasks: Optional[int] = Field(default=None, alias="max_concurrent_tasks")
    # Number of SLURM array elements to spread the outer loop over. Fewer elements
    # than outer combinations packs several combinations into each element, which is
    # how a sweep larger than MaxArraySize (or than a submit limit) gets submitted.
    # Omit to keep one array element per outer combination.
    num_array_jobs: Optional[int] = Field(default=None, ge=1, alias="num_array_jobs")

    model_config = {"extra": "allow", "populate_by_name": True}


def get_slurm_cpus_per_task(slurm: SlurmConfig) -> Optional[int]:
    """Reads 'cpus-per-task' from the SLURM block.

    SlurmConfig allows arbitrary SBATCH keys, so anything not declared as a field
    lands in model_extra under its literal JSON spelling. Returns None if unset.
    """
    extra = slurm.model_extra or {}
    raw = extra.get("cpus-per-task", extra.get("cpus_per_task"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_slurm_mem_mb(value: Any) -> Optional[int]:
    """Parses a SLURM memory value ('10G', '512M', '2048') into whole MB.

    A bare number is MB, matching SLURM's own convention. Returns None if the
    value cannot be interpreted.
    """
    if value is None:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT])?B?\s*", str(value), re.IGNORECASE)
    if not match:
        return None
    scale = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}
    return int(float(match.group(1)) * scale[(match.group(2) or "M").upper()])


def parse_slurm_time_seconds(value: Any) -> Optional[int]:
    """Parses a SLURM time limit into seconds.

    Accepts every format sbatch does: "MM", "MM:SS", "HH:MM:SS", "D-HH",
    "D-HH:MM" and "D-HH:MM:SS". Returns None for UNLIMITED/INFINITE or anything
    unparseable, so callers can treat "unknown" and "no limit" the same way.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"UNLIMITED", "INFINITE", "NONE"}:
        return None

    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        if not day_part.isdigit():
            return None
        days = int(day_part)
        # With a day part the remainder is HH[:MM[:SS]], never MM:SS.
        fields = text.split(":") if text else ["0"]
        if len(fields) > 3 or not all(f.isdigit() for f in fields):
            return None
        fields += ["0"] * (3 - len(fields))
        hours, minutes, seconds = (int(f) for f in fields)
    else:
        fields = text.split(":")
        if len(fields) > 3 or not all(f.isdigit() for f in fields):
            return None
        if len(fields) == 1:            # bare number is minutes
            hours, minutes, seconds = 0, int(fields[0]), 0
        elif len(fields) == 2:          # MM:SS
            hours, minutes, seconds = 0, int(fields[0]), int(fields[1])
        else:                           # HH:MM:SS
            hours, minutes, seconds = (int(f) for f in fields)

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


class TabularInnerLoop(BaseModel):
    type: Literal["tabular_file"]
    file_path: Optional[Path] = None
    delimiter: str = Field(default=" ")
    comment_prefix: Union[str, List[str]] = "#"  # a single prefix, or several, e.g. ["#", "*", "!"]
    args: List[TabularArgSpec] = []
    arg_names: Optional[List[str]] = None # Simpler specification for arg names without column mapping
    skip_blank_lines: bool = True

    # In case an old config uses 'inner_loop' as a dict without a 'type', we can migrate it to the new structure.
    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_inner_loop(cls, data):
        if not isinstance(data, dict):
            return data

        inner_loop = data.get("inner_loop")

        if isinstance(inner_loop, dict) and "type" not in inner_loop:
            inner_loop["type"] = "tabular_file"

        return data

    @field_validator("file_path")
    @classmethod
    def check_inner_file(cls, v: Optional[Path], info: ValidationInfo) -> Optional[Path]:
        if v is not None:
            v = resolve_config_path(v, info)
        # Reads 'check_files' from context, defaulting to True if not provided
        check_files = info.context.get("check_files", True) if info.context else True
        
        if check_files and v is not None and not v.is_file():
            raise ValueError(f"Inner loop file does not exist: '{v}'")
        return v
    
    @model_validator(mode="after")
    def migrate_arg_names(self) -> "TabularInnerLoop":
        """Converts 'arg_names' list into structured 'args' column specifications."""
        if not self.args and self.arg_names:
            self.args = [
                TabularArgSpec(arg_name=name, column=idx)
                for idx, name in enumerate(self.arg_names)
            ]
        if not self.args:
            raise ValueError("TabularInnerLoop requires 'args' (or simpler 'arg_names').")
        return self

    @property
    def arg_name_list(self) -> List[str]:
        """Convenience property for legends and validation checks."""
        return [spec.arg_name for spec in self.args]

# Tagged Union discriminator routes validation based on the 'type' field
InnerLoop = Annotated[
    Union[ExplicitLoop, TabularInnerLoop],
    Field(discriminator="type"),
]


class ExperimentConfig(BaseModel):
    result_database_path: str
    experiment_name: str
    slrm_output_dir: str
    tracking_args: Optional[Dict[str, str]] = None

    @field_validator("result_database_path")
    @classmethod
    def resolve_results_path(cls, v: str, info: ValidationInfo) -> str:
        # Must be absolute: the generated script changes into the executable's folder
        return str(resolve_config_path(v, info))

    @computed_field
    @property
    def checkpoint_dir(self) -> str:
        # Path() normalizes extra slashes automatically
        full_path = Path(self.result_database_path) / self.experiment_name / "checkpoints"
        return str(full_path)


class PatternSpec(BaseModel):
    """One value-extraction rule for 'collect', matched against each line of a run's output.

    'pattern' is a template such as 'Final results: {result} +/- {error}'; 'regex'
    is the escape hatch for lines a template cannot describe.
    """
    model_config = {"extra": "forbid"}

    pattern: Optional[str] = None
    regex: Optional[str] = None
    stream: Literal["out", "err", "any"] = "any"
    match: Literal["first", "last", "all"] = "last"

    @model_validator(mode="after")
    def compile_matcher(self) -> "PatternSpec":
        if (self.pattern is None) == (self.regex is None):
            raise ValueError("each collect pattern needs exactly one of 'pattern' or 'regex'.")
        pattern_matcher(self)  # raises ValueError naming the template if it is malformed
        return self

    @property
    def names(self) -> List[str]:
        return list(pattern_matcher(self)[1])

    @property
    def label(self) -> str:
        return self.pattern if self.pattern is not None else self.regex


class JsonSpec(BaseModel):
    model_config = {"extra": "forbid"}

    # Placeholders: any argument a run printed, plus {job_id}, {run} (or {indicator}), {exp_dir}
    path: str = DEFAULT_JSON_PATH

    @field_validator("path")
    @classmethod
    def resolve_json_path(cls, v: str, info: ValidationInfo) -> str:
        """A relative path is taken from the config file's folder; one starting with a
        placeholder (e.g. '{json}', '{exp_dir}/...') is resolved per run instead."""
        return v if v.startswith(("/", "{")) else str(resolve_config_path(v, info))


class CollectConfig(BaseModel):
    model_config = {"extra": "forbid", "populate_by_name": True}

    formats: List[Literal["sqlite", "csv"]] = Field(default_factory=lambda: ["sqlite"], min_length=1)
    patterns: List[PatternSpec] = Field(default_factory=list)
    # 'json' would shadow a BaseModel attribute, hence the alias. An explicit null turns JSON off.
    json_file: Optional[JsonSpec] = Field(default_factory=JsonSpec, alias="json")

    @field_validator("patterns", mode="before")
    @classmethod
    def expand_short_patterns(cls, value: Any) -> Any:
        """A bare string is shorthand for {pattern: <string>}."""
        if isinstance(value, list):
            return [{"pattern": v} if isinstance(v, str) else v for v in value]
        return value

    @field_validator("json_file", mode="before")
    @classmethod
    def expand_short_json(cls, value: Any) -> Any:
        """A bare string is shorthand for {path: <string>}."""
        return {"path": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def check_column_names(self) -> "CollectConfig":
        """Placeholder names become columns, so they must be unique and not reuse run-info columns."""
        seen: Dict[str, str] = {}
        for spec in self.patterns:
            for name in spec.names:
                if name in BUILTIN_COLUMNS:
                    raise ValueError(
                        f"placeholder '{name}' in pattern {spec.label!r} reuses a built-in column "
                        f"({', '.join(BUILTIN_COLUMNS)})."
                    )
                if name in seen:
                    raise ValueError(
                        f"placeholder '{name}' appears in both {seen[name]!r} and {spec.label!r}."
                    )
                seen[name] = spec.label
        return self


# --- 3. Root Application Schema ---

class AppConfig(BaseModel):
    loopQ: Optional[bool] = None  # Deprecated: the mode follows from which loop sections are present
    # Required; Optional only so a missing block gets the clear message in validate_required_sections
    execution: Optional[ExecutionConfig] = None
    slurm: Optional[SlurmConfig] = None
    inner_loop: Optional[InnerLoop] = None
    outer_loops: List[OuterLoopBlock] = []
    experiment: Optional[ExperimentConfig] = None
    args: Dict[str, Any] = {}
    collect: Optional[CollectConfig] = None  # What the 'collect' subcommand extracts per run
    # Names of arguments (defined in args, a loop or tracking_args) passed by position
    # instead of as flags, in command-line order
    positional_args: List[str] = Field(default_factory=list)
    positional_placement: Literal["last", "first"] = "last"  # after the flags, or before them

    @model_validator(mode="before")
    @classmethod
    def preprocess_legacy_inner_loop(cls, data: Any) -> Any:
        """Inject default 'type' into inner_loop dictionary before discriminator check."""
        if isinstance(data, dict):
            inner_loop = data.get("inner_loop")
            if isinstance(inner_loop, dict) and "type" not in inner_loop:
                # Modifying a shallow copy or in-place to set default discriminator tag
                data["inner_loop"]["type"] = "tabular_file"
        return data

    @model_validator(mode="before")
    @classmethod
    def reject_removed_inner_files(cls, data: Any) -> Any:
        """Rejects the withdrawn per-outer-value 'inner_files' mapping.

        Inner loops are now evaluated once in Python and shared across every outer
        combination, so a per-value file mapping cannot be honoured. Unknown keys are
        otherwise ignored, so without this the run would silently use the wrong
        parameter set instead of failing.
        """
        if isinstance(data, dict):
            if data.get("process_datasets"):
                raise ValueError(
                    "'process_datasets' (combining tabular datasets) is not part of SlurmDunk; "
                    "it remains available in Code-Orchestrator."
                )
            for idx, block in enumerate(data.get("outer_loops") or []):
                if isinstance(block, dict) and "inner_files" in block:
                    raise ValueError(
                        f"outer_loops[{idx}] uses 'inner_files', which is no longer supported. "
                        "Inner loop values are shared across all outer combinations; use a "
                        "separate config per inner file if they need to differ."
                    )
        return data

    @property
    def mode_info(self) -> Tuple[bool, int]:
        """
        Replaces determine_loop_q logic.
        Returns:
            (is_loop_active: bool, mode_index: int)
            - mode_index 0: Single run mode (no loops)
            - mode_index 1: Inner loop only
            - mode_index 2: Inner and outer loops
        """
        # 1. Explicit loopQ flag present
        if self.loopQ is not None:
            if not self.loopQ:
                return False, 0
            mode = 2 if self.outer_loops else 1
            return True, mode

        # 2. Implicit check based on presence of inner/outer blocks
        has_inner = self.inner_loop is not None
        has_outer = len(self.outer_loops) > 0

        if not has_inner and not has_outer:
            return False, 0

        mode = 2 if has_outer else 1
        return True, mode

    def argument_sources(self) -> Dict[str, List[str]]:
        """Maps every argument name, as written in the config, to where it is defined:
        'fixed' (args), 'outer', 'inner' or 'tracking'. A name defined in several
        places lists each of them."""
        sources: Dict[str, List[str]] = {}

        def add(name: Any, source: str) -> None:
            sources.setdefault(str(name), []).append(source)

        for key in self.args:
            add(key, "fixed")
        if isinstance(self.inner_loop, TabularInnerLoop):
            for name in self.inner_loop.arg_name_list:
                add(name, "inner")
        elif self.inner_loop is not None and getattr(self.inner_loop, "arg_name", None):
            add(self.inner_loop.arg_name, "inner")
        for block in self.outer_loops:
            if isinstance(block, TabularOuterLoop):
                for spec in block.args:
                    add(spec.arg_name, "outer")
            elif getattr(block, "arg_name", None):
                add(block.arg_name, "outer")
        if self.experiment and self.experiment.tracking_args:
            for key in self.experiment.tracking_args:
                add(key, "tracking")
        return sources

    def argument_names(self) -> Set[str]:
        """All argument keys across all configuration sections, as written in the config;
        passed_flags turns them into the flag spellings the executable actually receives."""
        return set(self.argument_sources())

    def passed_flags(self) -> List[str]:
        """Every flag spelling the generated script passes to the executable.

        Mirrors how the script builds its command lines: format_cli_args for args and
        loop arguments ('x' -> '-x', 'beta' -> '--beta'), and always '--<key>' for
        tracking args (build_experiment_strings). Positional arguments have no flag.
        """
        positional = set(self.positional_args)
        flags = {format_cli_arg(name) for name in self.argument_names() if name not in positional}
        if self.experiment and self.experiment.tracking_args:
            flags.update(f"--{key}" for key in self.experiment.tracking_args if key not in positional)
        return sorted(flags)

    @model_validator(mode="after")
    def validate_positional_args(self) -> "AppConfig":
        """Each positional name must have exactly one definition and a value that can stand alone."""
        if not self.positional_args:
            return self
        sources = self.argument_sources()
        seen = set()
        for name in self.positional_args:
            if name.startswith("-"):
                raise ValueError(
                    f"positional_args: '{name}' has leading dashes, but a positional argument has no flag; "
                    "list it (and define it) without them."
                )
            if name in seen:
                raise ValueError(f"positional_args lists '{name}' more than once.")
            seen.add(name)
            defined = sources.get(name, [])
            if not defined:
                dashed = [k for k in (f"--{name}", f"-{name}") if k in sources]
                hint = f" It is defined as '{dashed[0]}'; drop the dashes there." if dashed else ""
                raise ValueError(
                    f"positional_args: '{name}' is not defined in 'args', a loop or 'tracking_args'.{hint}"
                )
            if len(defined) > 1:
                raise ValueError(
                    f"positional_args: '{name}' is defined more than once ({', '.join(defined)}); "
                    "a positional argument takes exactly one value per run."
                )
            if defined == ["fixed"]:
                check_positional_value(name, self.args[name], "in 'args'")
        return self

    @property
    def mode_index(self) -> int:
        """Helper property to access mode_index (0, 1, or 2) directly."""
        return self.mode_info[1]

    @property
    def is_loop_mode(self) -> bool:
        """Helper property to check if loop mode is active."""
        return self.mode_info[0]

    @model_validator(mode="after")
    def validate_required_sections(self) -> "AppConfig":
        if self.execution is None:
            raise ValueError("Missing 'execution' configuration block.")
        if self.slurm is None:
            raise ValueError("Missing 'slurm' configuration block.")
        return self

    @model_validator(mode="after")
    def validate_multithreading_level(self) -> "AppConfig":
        """Checks that one run actually fits inside the requested allocation."""
        level = self.execution.multithreading_level
        if level is None:
            return self

        cpus = get_slurm_cpus_per_task(self.slurm)
        if cpus is not None and level > cpus:
            raise ValueError(
                f"execution.multithreading_level ({level}) exceeds slurm 'cpus-per-task' "
                f"({cpus}); a single run would not fit in the allocation."
            )
        return self

    @model_validator(mode="after")
    def validate_mode_integrity(self) -> "AppConfig":
        """Enforces rules based on determined mode_index."""
        is_loop, mode = self.mode_info

        if is_loop:
            if self.inner_loop is None:
                raise ValueError("Missing 'inner_loop' section (required when loop mode is active).")
        else:
            # Single run mode
            if self.inner_loop is not None:
                raise ValueError("'inner_loop' should not be present when loop mode is disabled.")

        return self


# --- 4. Replacement Validation Entrypoint ---

# Spellings that still work but are going away in 1.0; each warning is printed once per config
_WARNED: set = set()


def deprecation_warnings(cfg: Any) -> List[str]:
    """Deprecated spellings used in a raw config, as one message each."""
    messages = []
    if not isinstance(cfg, dict):
        return messages
    if "loopQ" in cfg:
        messages.append("'loopQ' is deprecated and will be removed in 1.0: the mode follows from "
                        "which loop sections are present.")
    execution = cfg.get("execution")
    if isinstance(execution, dict) and "language" in execution:
        messages.append("'execution.language' is deprecated; use 'execution.interpreter'.")
    experiment = cfg.get("experiment")
    tracking = experiment.get("tracking_args") if isinstance(experiment, dict) else None
    if isinstance(tracking, dict) and any("{__indicator}" in str(v) for v in tracking.values()):
        messages.append("'{__indicator}' in tracking_args is deprecated; use '{run}'.")
    return messages


def validate_config(cfg: dict, config_path: str, dryrunQ: bool) -> AppConfig:
    """Validates structure, types, and file existence using Pydantic.

    Relative paths in the config are taken from the config file's folder.
    """
    context = {"check_files": not dryrunQ, "base_dir": str(Path(config_path).resolve().parent)}
    try:
        validated_cfg = AppConfig.model_validate(cfg, context=context)
        for message in deprecation_warnings(cfg):
            if (str(config_path), message) not in _WARNED:
                _WARNED.add((str(config_path), message))
                print(f"[WARNING] {config_path}: {message}", file=sys.stderr)
        return validated_cfg
    except ValidationError as e:
        print(f"\n[CONFIG ERROR] Validation failed for '{config_path}':", file=sys.stderr)
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            print(f"  - [{loc}]: {err['msg']}", file=sys.stderr)
        sys.exit(1)
