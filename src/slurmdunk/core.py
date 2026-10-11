import csv
import json
import yaml
from datetime import datetime
import shutil
import sys
from pathlib import Path
import subprocess
import re
import shlex
from typing import Any, Collection, Dict, Optional, List, Set, Tuple

# Required for Pydantic based validation of input config file
from .config_schema import (
    validate_config, AppConfig, ExperimentConfig, SlurmConfig, ExecutionConfig,
    TabularInnerLoop, TabularOuterLoop, get_slurm_cpus_per_task, parse_slurm_mem_mb,
    parse_slurm_time_seconds,
)

# Required for constructing mode-specific portion of SLURM scripts
from .mode_builders import (
    build_single_mode, build_inner_loop_mode, build_job_array_mode, build_exec_command, _evaluate_tabular_rows,
)

# Required for evaluating process_datasets 'cuts' expressions
from .arg_transform import _evaluate_cut

# Utilities
from . import __version__
from .utils import (
    PositionalValueError, bash_dq_escape, format_cli_arg, format_cli_args, split_positional,
    tracking_template_to_bash,
)

def get_dict_from_config_file(config_path) -> dict:
  config_file = Path(config_path)
  if not config_file.is_file():
    print(
        f"[ERROR] Configuration file '{config_path}' not found.",
        file=sys.stderr,
    )
    sys.exit(1)

  suffix = config_file.suffix.lower()

  with open(config_file, "r", encoding="utf-8") as f:
    try:
      if suffix == ".json":
        cfg = json.load(f)
      elif suffix in (".yaml", ".yml"):
        cfg = yaml.safe_load(f)
      else:
        # Fallback for unrecognized extensions: YAML parses valid JSON natively
        cfg = yaml.safe_load(f)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
      print(
          f"[ERROR] Failed to parse configuration file '{config_path}': {e}",
          file=sys.stderr,
      )
      sys.exit(1)

  if not isinstance(cfg, dict):
    print(
        f"[ERROR] Configuration in '{config_path}' must be a valid YAML or JSON file interpretable as a key-value mapping"
        " (dictionary).",
        file=sys.stderr,
    )
    sys.exit(1)

  return cfg


def build_experiment_strings(
    experiment: Optional[ExperimentConfig], mode_index, positional: Collection[str] = ()
) -> Tuple[str, str, str]:
    """
    Builds the environment block and argument block for experiment tracking.
    Returns (exp_name, exp_env_block, exp_args_block).

    Tracking args named in positional are left out of exp_args: the generated
    script passes them by position instead (see mode_builders).
    """
    if experiment is None:
        exp_env_block = """
# Absolute so it still resolves after the script cd's to the executable directory
export CHECKPOINT_DIR="$(pwd)/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
"""
        exp_args_block = '\n    exp_args=""'
        return "noexp", exp_env_block, exp_args_block

    db_path = experiment.result_database_path
    exp_name = experiment.experiment_name
    slrm_output_dir_name = experiment.slrm_output_dir

    job_id_var = "SLURM_ARRAY_JOB_ID" if mode_index == 2 else "SLURM_JOB_ID"

    exp_env_block = f"""
# =========================================================================
# EXPERIMENT TRACKING
# =========================================================================
export RESULT_DATABASE_PATH="{db_path}"
export EXPERIMENT_NAME="{exp_name}"
export SLRM_OUTPUT_DIR="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/{slrm_output_dir_name}"

# Safe directory creation for the specific job
export JOB_ID=${{{job_id_var}:-$$}}
export SAVE_DIR="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/$JOB_ID"
export CHECKPOINT_DIR="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/checkpoints"

mkdir -p "$SAVE_DIR"
mkdir -p "$CHECKPOINT_DIR"
"""

    tracking_args = experiment.tracking_args
    arg_components = []
    if tracking_args is not None:
        for key, val_template in tracking_args.items():
            if key in positional:
                continue
            formatted_val = tracking_template_to_bash(val_template)
            arg_components.append(f'--{key} \\"{formatted_val}\\"')
    tracking_args_str = " ".join(arg_components)

    exp_args_block = f"""
    exp_args=""
    if [ -n "${{SAVE_DIR:-}}" ]; then
        exp_args="{tracking_args_str}"
    fi"""
    return exp_name, exp_env_block, exp_args_block

def setup_experiment_directories(
    experiment: Optional[ExperimentConfig],
    slurm: SlurmConfig,
    config_file: Path,
    mode_index: int,
    dryrun_q: bool,
    timestamp: str,
) -> Optional[Path]:
    """
    Creates the SLURM output directory and updates slurm.output on the model.

    Returns the path the run's backup directory *should* take, without creating
    it. Creation is deferred to write_script, so a run that turns out to have
    nothing to submit - or whose script fails its syntax check - leaves no
    stray timestamped directory behind.
    """
    if experiment is None:
        return

    db_path = experiment.result_database_path
    exp_name = experiment.experiment_name
    slrm_output_dir_name = experiment.slrm_output_dir

    full_slrm_output_dir = f"{db_path}/{exp_name}/{slrm_output_dir_name}"
    # Mutates the Pydantic model directly
    slurm.output = f"{full_slrm_output_dir}/{exp_name}_%j.out"
    if mode_index == 2:
        # For array mode, we want to include the indicator in the output filename
        slurm.output = f"{full_slrm_output_dir}/{exp_name}_%A_%a.out"
    try:
        Path(full_slrm_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ensured SLURM output directory exists: {full_slrm_output_dir}")
    except Exception as e:
        print(f"[WARNING] Could not create SLRM_OUTPUT_DIR '{full_slrm_output_dir}': {e}", file=sys.stderr)

    if dryrun_q:
        return None
    return Path(db_path) / exp_name / timestamp
  
def _get_generator_git_commit() -> str:
    """Retrieves the git commit of the SlurmDunk checkout, or the package version when
    it is installed rather than run from a git checkout."""
    try:
        file_dir = Path(__file__).resolve().parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=file_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return f"none (slurmdunk {__version__}, not a git checkout)"

def build_common_header(
    slurm: SlurmConfig,
    exec_cfg: ExecutionConfig,
    exp_env_block: str,
    timestamp: str = "",
    backup_dir: Optional[Path] = None,
    config_tag: str = "",
    arg_flags: Optional[List[str]] = None,
) -> str:
    """Generates the SBATCH headers, environment modules, export variables, and print helper function.

    arg_flags are the exact flag spellings passed to the executable (see
    get_passed_flags); print_args only treats these as flags.
    """
    generator_commit = _get_generator_git_commit()

    # Dump SlurmConfig using aliases (e.g. 'job-name') and excluding None values
    slurm_dict = slurm.model_dump(by_alias=True, exclude_none=True)

    # Exclude internal configuration fields not meant for SBATCH
    slurm_dict.pop("max_concurrent_tasks", None)
    slurm_dict.pop("num_array_jobs", None)

    # The comment carries the config tag that the duplicate-submission check
    # matches against, so a user-supplied one cannot be kept alongside it.
    if config_tag and slurm_dict.pop("comment", None) is not None:
        print(
            "[WARNING] slurm 'comment' is reserved for SlurmDunk's config tag; "
            "the configured value is ignored.",
            file=sys.stderr,
        )

    slurm_header = "".join(f'#SBATCH --{k}={v}\n' for k, v in slurm_dict.items())
    if config_tag:
        slurm_header += f'#SBATCH --comment="{config_tag}"\n'

    modules = exec_cfg.modules
    module_load_block = (
        "\n# Load Required Environment Modules\n" + "\n".join(f"module load {m}" for m in modules)
        if modules else "# No environment modules specified"
    )

    # Exported before env_vars so a config can set e.g. JULIA_NUM_THREADS to
    # "$MULTITHREADING_LEVEL" and get the per-run thread count.
    level_block = (
        f"\n# Cores used by one invocation of the executable\n"
        f"export MULTITHREADING_LEVEL={exec_cfg.multithreading_level}"
        if exec_cfg.multithreading_level is not None else ""
    )

    env_vars = exec_cfg.env_vars
    env_var_block = (
        "\n# Set Environment Variables\n" + "\n".join(f'export {k}={v}' for k, v in env_vars.items())
        if env_vars else "# No environment variables specified"
    )

    # The disown is required: the script later uses bare `wait` (and `wait -n` in
    # parallel mode) to flush its log pipes, and those also wait on any background
    # process the preamble left running, e.g. a GPU monitor that the epilogue is
    # meant to kill. The job would then hang until walltime. Disowned processes
    # drop out of bash's job table but keep their PIDs, so `kill $PID` still works,
    # and SLURM kills whatever is left when the script exits.
    preamble = exec_cfg.preamble
    preamble_block = (
        "\n# User-Supplied Preamble\n" + preamble.rstrip("\n")
        + "\n# Keep background processes started above from blocking the script's waits\n"
        + "disown -a"
        if preamble else "# No preamble specified"
    )

    # Space-delimited on both ends so print_args can test membership with *" $1 "*.
    flags_list = shlex.quote(" " + " ".join(arg_flags or []) + " ")
    print_args_def = f"""
# Every flag spelling passed to the executable. print_args needs the exact list:
# a single-letter flag such as -x and a negative value such as -3 look alike.
PRINT_ARGS_FLAGS={flags_list}
""" + """
# Function to parse and display command-line arguments to SLURM output
print_args() {
    local ind="$1"
    shift
    local args_str="$*"
    if [ -z "$args_str" ]; then
        return
    fi
    eval "set -- $args_str"
    while [ $# -gt 0 ]; do
        if [[ "$PRINT_ARGS_FLAGS" == *" $1 "* ]]; then
            local param_name="${1#-}"
            param_name="${param_name#-}"
            if [ $# -gt 1 ] && [[ "$PRINT_ARGS_FLAGS" != *" $2 "* ]]; then
                echo "[${ind}_args] ${param_name}: $2"
                shift 2
            else
                echo "[${ind}_args] ${param_name}: "
                shift 1
            fi
        else
            shift 1
        fi
    done
}
"""
    # Echoed (not just commented) so the generation timestamp lands in the SLURM
    # output itself, tying a job back to the config/script snapshot that produced
    # it. Carries no '_out]' or 'Job duration:' text, so --collect ignores it.
    provenance_lines = [f'echo "[GENERATION] Script generated at: {timestamp}"'] if timestamp else []
    if backup_dir is not None:
        provenance_lines.append(f'echo "[GENERATION] Config/script snapshot: {backup_dir}"')
    provenance_block = "\n".join(provenance_lines)

    return f"""#!/bin/bash
{slurm_header}

# =========================================================================
# GENERATOR METADATA
# Generated by SlurmDunk commit: {generator_commit}
# Generation timestamp: {timestamp}
# =========================================================================
{provenance_block}

{exp_env_block}
{module_load_block}
{level_block}
{env_var_block}
{preamble_block}
{print_args_def}"""


def build_common_footer(exec_cfg: ExecutionConfig) -> str:
    """Generates the user-supplied epilogue, appended after the last executable run.

    Kept out of build_common_header: array mode rewrites the header line by line,
    and the epilogue has to come after the mode-specific body anyway. The trailing
    newline is forced so an epilogue ending in a heredoc terminator stays valid.
    """
    epilogue = exec_cfg.epilogue
    return (
        "\n# User-Supplied Epilogue\n" + epilogue.rstrip("\n") + "\n"
        if epilogue else "\n# No epilogue specified\n"
    )


def warn_about_concurrency(config: AppConfig) -> None:
    """Flags configurations whose thread settings will oversubscribe the node.

    Thread counts stay the user's responsibility (nothing is rewritten), but an
    env var still pinned to $SLURM_CPUS_PER_TASK while the inner loop runs N ways
    concurrently means N x the cores are requested, which is slower than running
    sequentially. Warn rather than block.
    """
    level = config.execution.multithreading_level
    if level is None:
        return

    offenders = [
        k for k, v in config.execution.env_vars.items()
        if "SLURM_CPUS_PER_TASK" in str(v)
    ]
    if offenders:
        print(
            f"[WARNING] {', '.join(offenders)} still reference(s) $SLURM_CPUS_PER_TASK while "
            f"multithreading_level={level} is set.",
            file=sys.stderr,
        )
        print(
            "[WARNING] Each concurrent run would request the whole allocation. Set these to "
            "$MULTITHREADING_LEVEL (exported by the generated script) or a literal value.",
            file=sys.stderr,
        )

    cpus = get_slurm_cpus_per_task(config.slurm)
    if cpus:
        njobs = max(1, cpus // level)
        mem_mb = parse_slurm_mem_mb((config.slurm.model_extra or {}).get("mem"))
        detail = ""
        if mem_mb and njobs > 1:
            detail = f"; --mem is shared, leaving ~{mem_mb // njobs} MB per concurrent run"
        print(f"[INFO] Inner loop will run up to {njobs} concurrent run(s){detail}.")


def get_partition_max_time(partition: str) -> Tuple[Optional[int], Optional[str]]:
    """Looks up a partition's MaxTime via scontrol.

    Returns (max_seconds, error). max_seconds is None when the limit is unknown
    or UNLIMITED. error is set only when the partition genuinely does not exist;
    when scontrol itself is unavailable both are None so the caller skips the
    check (e.g. a --dryrun off the cluster).
    """
    try:
        result = subprocess.run(
            ["scontrol", "show", "partition", partition],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None  # no scontrol here: nothing to check against

    if result.returncode != 0:
        return None, f"partition '{partition}' does not exist on this cluster"

    match = re.search(r"MaxTime=(\S+)", result.stdout)
    if not match:
        return None, None
    return parse_slurm_time_seconds(match.group(1)), None


def validate_partition_time_limit(config: AppConfig) -> None:
    """Rejects a requested time limit the partition would refuse.

    Mirrors the executable argument check: a violation is a hard error rather
    than a warning, since sbatch would reject the submission anyway. Runs at
    generation time because the default workflow generates a script and submits
    it manually later, so a submit-time-only check would miss most usage.
    """
    partition = config.slurm.partition
    requested = parse_slurm_time_seconds(config.slurm.time)

    max_seconds, error = get_partition_max_time(partition)
    if error is not None:
        print(f"\n[ERROR] Partition validation failed: {error}.", file=sys.stderr)
        print("[ERROR] Use --notimecheck to skip this check.", file=sys.stderr)
        sys.exit(1)

    if max_seconds is None or requested is None:
        return

    if requested > max_seconds:
        print(
            f"\n[ERROR] Requested time '{config.slurm.time}' exceeds the limit of "
            f"partition '{partition}'.",
            file=sys.stderr,
        )
        print(
            f"[ERROR] Partition MaxTime is {format_seconds_as_slurm_time(max_seconds)}; "
            f"requested {format_seconds_as_slurm_time(requested)}.",
            file=sys.stderr,
        )
        print("[ERROR] Use --notimecheck to skip this check.", file=sys.stderr)
        sys.exit(1)


# Written into each generated script's SBATCH comment, so a queued job can be
# traced back to the config file it was generated from.
CONFIG_TAG_PREFIX = "slurmdunk-config:"
# Written by the framework before it became SlurmDunk; still matched, so jobs
# submitted with the old tool are caught by the duplicate-submission check.
LEGACY_CONFIG_TAG_PREFIXES = ("orchestrator-config:",)


def get_config_tag(config_path) -> str:
    """Identifies a config in SLURM's comment field by its resolved absolute path."""
    return f"{CONFIG_TAG_PREFIX}{Path(config_path).resolve()}"


def find_live_jobs_for_config(config_path) -> Tuple[Optional[Dict[str, Set[str]]], Optional[str]]:
    """Looks up the user's queued/running jobs that were generated from this config.

    Returns (jobs, error). jobs maps base job ID -> set of states, so an array
    job appears once however many of its tasks are listed. Both are None when
    squeue is unavailable (e.g. off the cluster); error is set when squeue exists
    but the queue could not be read.
    """
    resolved = Path(config_path).resolve()
    tags = {f"{prefix}{resolved}" for prefix in (CONFIG_TAG_PREFIX,) + LEGACY_CONFIG_TAG_PREFIXES}
    try:
        result = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%F|%T|%k"],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        return None, None  # no squeue here: nothing to check against
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)

    if result.returncode != 0:
        return None, result.stderr.strip() or f"squeue exited with code {result.returncode}"

    jobs: Dict[str, Set[str]] = {}
    for line in result.stdout.splitlines():
        # The comment is the last field, so a '|' inside it cannot shift the others
        parts = line.strip().split("|", 2)
        if len(parts) == 3 and parts[2] in tags:
            jobs.setdefault(parts[0], set()).add(parts[1])
    return jobs, None


def check_duplicate_submission(config_path) -> bool:
    """Returns True if the config must be skipped because jobs generated from it
    are still in the queue.

    Checkpointing only skips work that has finished; queued or running tasks
    would be submitted a second time. Runs at generation time for the same
    reason as the partition time check: the script is often submitted by hand.
    A queue that cannot be read also skips the config, since the safeguard
    should not wave a submission through unchecked.
    """
    jobs, error = find_live_jobs_for_config(config_path)
    if error is not None:
        print(f"[ERROR] Skipping {config_path}: could not query the SLURM queue: {error}", file=sys.stderr)
        print("[ERROR] Retry, or pass --nodupcheck to skip the duplicate-submission check.", file=sys.stderr)
        return True

    if jobs is None:
        print("[WARNING] 'squeue' not found; skipping the duplicate-submission check.", file=sys.stderr)
        return False

    if not jobs:
        return False

    print(
        f"[ERROR] Skipping {config_path}: job(s) generated from this config are still in the queue:",
        file=sys.stderr,
    )
    for job_id, states in sorted(jobs.items()):
        print(f"[ERROR]   {job_id} ({', '.join(sorted(states))})", file=sys.stderr)
    print(
        "[ERROR] Wait for them to finish (checkpointing then skips completed work), scancel them, "
        "or pass --nodupcheck.",
        file=sys.stderr,
    )
    return True


def format_seconds_as_slurm_time(total_seconds: int) -> str:
    """Renders seconds as SLURM's D-HH:MM:SS, dropping the day part when zero."""
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    stamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{stamp}" if days else stamp


def check_generated_script(script_content: str) -> None:
    """Syntax-checks the generated bash and guards SLURM's script size limit."""
    size = len(script_content.encode("utf-8"))
    if size > 120_000:
        print(
            f"[ERROR] Generated script is {size} bytes; SLURM rejects scripts over 131072. "
            "Reduce the parameter space or split the config.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result = subprocess.run(
            ["bash", "-n"], input=script_content, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[WARNING] Could not syntax-check the generated script: {e}", file=sys.stderr)
        return

    if result.returncode != 0:
        print("[ERROR] Generated script failed 'bash -n' syntax check:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)


def write_script(
    script_content: str,
    mode_prefix: str,
    exp_name: str,
    timestamp: str,
    backup_dir: Optional[Path] = None,
    config_file: Optional[Path] = None,
) -> Path:
    """
    Writes the script content to a file and prints the location.
    mode_prefix is one of 'single', 'inner', 'array'.

    Also snapshots the config and the script into the backup directory, which is
    created here rather than earlier so that it only ever exists for a run that
    actually produced a script.
    """
    check_generated_script(script_content)

    script_file = Path(f"submit_{mode_prefix}_{exp_name}_{timestamp}.sh")
    script_file.write_text(script_content)
    print(f"Generated {mode_prefix}-run SLURM script: {script_file}")

    if backup_dir is not None:
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            if config_file is not None:
                shutil.copy2(config_file, backup_dir / config_file.name)
                print(f"[INFO] Copied config file to: {backup_dir / config_file.name}")
            shutil.copy2(script_file, backup_dir / script_file.name)
            print(f"[INFO] Copied SLURM script to: {backup_dir / script_file.name}")
        except Exception as e:
            print(f"[WARNING] Could not populate backup directory '{backup_dir}': {e}", file=sys.stderr)

    return script_file


def discard_backup_dir(backup_dir: Optional[Path]) -> None:
    """Removes a run's backup directory after a submission that produced no job.

    Only ever called for a directory this run created (<db>/<exp>/<timestamp>),
    so there is nothing here that could reach pre-existing results.
    """
    if backup_dir is None or not backup_dir.is_dir():
        return
    try:
        shutil.rmtree(backup_dir)
        print(f"[INFO] Removed backup directory for the failed submission: {backup_dir}")
    except Exception as e:
        print(f"[WARNING] Could not remove backup directory '{backup_dir}': {e}", file=sys.stderr)



def generate_slurm_script(
    config_path, dryrunQ, checktimeQ: bool = True
) -> Tuple[Optional[Path], Optional[Path]]:
    """Generates the submission script. Returns (script_path, backup_dir).

    script_path is None when there is nothing left to submit. backup_dir is
    returned so the caller can discard the snapshot if submission then fails.
    """
    cfg = get_dict_from_config_file(config_path)
    config_file = Path(config_path)

    # One timestamp for the whole run: it names the backup directory, names the
    # script, and is echoed into the SLURM output, so all three always agree.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Validate & convert raw dict to typed AppConfig object
    config: AppConfig = validate_config(cfg, config_path, dryrunQ)
    print(f"[SUCCESS] Config validation passed for '{config_path}'.")

    if config.process_datasets:
        print(
            f"[ERROR] '{config_path}' has 'process_datasets: true'. Use the 'process' "
            "subcommand instead of 'generate' (or the bare invocation, which defaults "
            "to 'generate').",
            file=sys.stderr,
        )
        sys.exit(1)

    warn_about_concurrency(config)
    if checktimeQ:
        validate_partition_time_limit(config)

    # 2. Extract execution metadata cleanly via dot-notation
    exec_cfg = config.execution
    slurm_cfg = config.slurm
    experiment_cfg = config.experiment
    max_concurrent = slurm_cfg.max_concurrent_tasks  # Defined on SlurmConfig model

    # 3. Filesystem setup (directories + config backup, function returns backup_dir path if created)
    backup_dir = setup_experiment_directories(
        experiment_cfg, slurm_cfg, config_file, config.mode_index, dryrunQ, timestamp
    )

    # 4. Build experiment strings & execution context
    exp_name, exp_env_block, exp_args_block = build_experiment_strings(
        experiment_cfg, config.mode_index, config.positional_args
    )
    common_header = build_common_header(
        slurm_cfg, exec_cfg, exp_env_block, timestamp, backup_dir, get_config_tag(config_file),
        get_passed_flags(config),
    )

    exec_path = Path(exec_cfg.executable).resolve()
    exec_dir = exec_path.parent
    flag_args, fixed_positional = split_positional(config.args, config.positional_args)
    fixed_args_str = format_cli_args(flag_args)
    flags_str = " ".join(exec_cfg.flags)
    # Fixed positional values belong to the checkpoint signature as fixed flags do.
    # Empty without positional args, so existing configs keep their hashes.
    fixed_positional_str = " ".join(
        str(fixed_positional[name]) for name in config.positional_args if name in fixed_positional
    )
    exec_sig_components = [exec_cfg.interpreter, flags_str, str(exec_path), fixed_args_str]
    exec_sig_str = " ".join(p for p in exec_sig_components + [fixed_positional_str] if p)
    # The same signature for the script's EXEC_SIG="..." line. Positional values may hold
    # spaces, quotes or $, so they are escaped and bash recovers exactly exec_sig_str;
    # the other parts are emitted as before, keeping existing scripts byte-identical.
    exec_sig_bash = " ".join(p for p in exec_sig_components + [bash_dq_escape(fixed_positional_str)] if p)

    checkpoint_dir = (
        Path(experiment_cfg.checkpoint_dir) if experiment_cfg else Path("checkpoints").resolve()
    )

    ctx = {
        "exec_cfg": exec_cfg,
        "checkpoint_dir": checkpoint_dir,
        "exec_path": exec_path,
        "exec_dir": exec_dir,
        "flags_str": flags_str,
        "fixed_args_str": fixed_args_str,
        "exec_sig_str": exec_sig_str,
        "exec_sig_bash": exec_sig_bash,
        "exp_args_block": exp_args_block,
        "common_header": common_header,
        "common_footer": build_common_footer(exec_cfg),
    }

    # 5. Clean Mode Routing via computed config properties
    mode_index = config.mode_index

    # Loop values are only known here, so a value that cannot be positional (true/false,
    # null, empty) is reported now, naming the argument and the row, before anything is written.
    try:
        if mode_index == 0:
            # Mode 0: Single Run
            script_content = build_single_mode(ctx, config)
            if script_content is None:
                return None, None
            return write_script(
                script_content, "single", exp_name, timestamp, backup_dir, config_file
            ), backup_dir

        elif mode_index == 1:
            # Mode 1: Inner Loop Only
            script_content = build_inner_loop_mode(ctx, config)
            if script_content is None:
                return None, None
            return write_script(
                script_content, "inner", exp_name, timestamp, backup_dir, config_file
            ), backup_dir

        else:
            # Mode 2: Array Mode (Outer + Inner Loops)
            script_content, total_tasks = build_job_array_mode(ctx, config, max_concurrent)
            if script_content is None:
                return None, None
            script_file = write_script(
                script_content, "array", exp_name, timestamp, backup_dir, config_file
            )
            print(f"({total_tasks} tasks)")
            return script_file, backup_dir
    except PositionalValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

def extract_config_arg_names(config: AppConfig) -> Set[str]:
    """Collects all argument keys across all configuration sections in AppConfig.

    These are the names as written in the config; get_passed_flags turns them
    into the flag spellings the executable actually receives.
    """
    return set(config.argument_sources())


def get_passed_flags(config: AppConfig) -> List[str]:
    """Every flag spelling the generated script passes to the executable.

    Mirrors how the script builds its command lines: format_cli_args for args and
    loop arguments ('x' -> '-x', 'beta' -> '--beta'), and always '--<key>' for
    tracking args (build_experiment_strings). Positional arguments have no flag.
    """
    positional = set(config.positional_args)
    flags = {format_cli_arg(name) for name in extract_config_arg_names(config) if name not in positional}
    if config.experiment and config.experiment.tracking_args:
        flags.update(f"--{key}" for key in config.experiment.tracking_args if key not in positional)
    return sorted(flags)


def count_usage_positionals(help_text: str) -> Optional[int]:
    """Best-effort count of the positional arguments in a usage line.

    Understands the 'usage: prog [-h] [--x X] infile outfile' form printed by
    argparse and ArgParse.jl: bracketed (optional) groups are dropped, as is the
    value following a required option; the remaining bare words are positionals.
    Returns None when there is no usage line.
    """
    lines = help_text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().lower().startswith("usage:")), None)
    if start is None:
        return None
    usage = [lines[start].strip()[len("usage:"):]]
    for line in lines[start + 1:]:
        if not line.strip() or not line[:1].isspace():  # wrapped usage lines are indented
            break
        usage.append(line.strip())

    text = " ".join(usage)
    while True:  # innermost brackets first, so nested optional groups disappear too
        stripped = re.sub(r"\[[^\[\]]*\]", " ", text)
        if stripped == text:
            break
        text = stripped

    count, after_option = 0, False
    for token in text.split()[1:]:  # [0] is the program name
        if after_option:
            after_option = False
        elif token.startswith("-"):
            after_option = True
        elif token != "...":
            count += 1
    return count


def validate_script_args(config: AppConfig) -> Tuple[bool, List[str]]:
    """Validates if arguments defined in the AppConfig instance are supported
    by the target executable via --help.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_unsupported_flags)
    """
    exec_cfg = config.execution
    executable_path = str(exec_cfg.executable)
    modules = exec_cfg.modules

    # The exact spellings the executable receives ('-x', '--beta'), shared with print_args
    flags_to_check = get_passed_flags(config)
    if not flags_to_check:
        return True, []

    # Build bash command to load modules and invoke help flag
    module_cmds = [f"module load {m}" for m in modules]
    # Built like the real run command, which leaves out a missing interpreter, so a
    # directly executable program is called as '<executable> --help'
    exec_cmd = build_exec_command(exec_cfg, "", "", "--help")
    full_cmd = " && ".join(module_cmds + [exec_cmd])

    try:
        result = subprocess.run(
            ["bash", "-c", full_cmd],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        help_output = result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr_msg = e.stderr if hasattr(e, "stderr") else str(e)
        raise RuntimeError(
            f"Failed to execute '--help' check on '{executable_path}':\n{stderr_msg}"
        ) from e

    # Extract all supported option flags from --help text
    raw_found_flags = set(re.findall(r"(?<!\w)(-[a-zA-Z0-9_|-]+)", help_output))

    supported_flags: Set[str] = set()
    for flag_group in raw_found_flags:
        for flag in re.split(r"[,|/]", flag_group):
            clean_flag = flag.strip()
            supported_flags.add(clean_flag)

    # Accept '_' and '-' interchangeably within a name (--batch_size vs --batch-size).
    # Only the name is rewritten: swapping the leading dashes too would turn
    # '--batch-size' into '__batch_size', which never matches.
    unsupported = []
    for flag in flags_to_check:
        name = flag.lstrip("-")
        dashes = flag[: len(flag) - len(name)]
        variants = {flag, dashes + name.replace("_", "-"), dashes + name.replace("-", "_")}

        if not (variants & supported_flags):
            unsupported.append(flag)

    # Positional arguments have no flag to look for; compare their number with the
    # usage line instead. Usage formats vary, so a mismatch only warns.
    if config.positional_args:
        found = count_usage_positionals(help_output)
        expected = len(config.positional_args)
        if found is not None and found != expected:
            print(
                f"[WARNING] The executable's usage line lists {found} positional argument(s), but "
                f"positional_args has {expected} ({', '.join(config.positional_args)}).",
                file=sys.stderr,
            )

    return len(unsupported) == 0, unsupported

def submit_slurm_script(
    script_path: Path,
    config: AppConfig,
    checkargs_q: bool,
    backup_dir: Optional[Path] = None,
) -> None:
    """Submits the generated script to SLURM.

    If submission was attempted but produced no job - a rejected sbatch, a
    missing sbatch, or arguments that failed validation before sbatch ran - the
    run's backup directory is discarded, so only snapshots corresponding to a
    real job survive. A run generated without --submit keeps its snapshot, since
    it is still waiting to be submitted by hand.
    """

    if checkargs_q:
        print("\n[INFO] Validating arguments passed to executable...")
        print("[INFO] This may take some time. Use --noargcheck to disable argument checking.")

        validation_result, failed_args = validate_script_args(config)
        if not validation_result:
            print("\n[ERROR] Argument validation failed!", file=sys.stderr)
            print(f"[ERROR] Offending args: {' '.join(failed_args)}", file=sys.stderr)
            discard_backup_dir(backup_dir)
            sys.exit(1)
        else:
            print("\n[SUCCESS] Argument validation succeeded!")
    
    print(f"\n[INFO] Submitting {script_path} to SLURM...")
    try:
        # Calls sbatch. The embedded checkpoint logic handles skips securely.
        result = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] SLURM Submission failed:\n{e.stderr}", file=sys.stderr)
        discard_backup_dir(backup_dir)
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 'sbatch' command not found. Are you on a SLURM cluster node?", file=sys.stderr)
        discard_backup_dir(backup_dir)
        sys.exit(1)

def process_datasets(config_path: str) -> None:
    """Combines every 'outer_loops' tabular source into one row-wise concatenated table.

    Each source's columns are extracted in its declared 'args' order (validated to match
    in count across all sources), the sources are concatenated row-wise, then 'process'
    optionally renames the columns and filters rows via 'cuts' before the result is
    written to 'output_file_path' as CSV.

    Unlike 'generate', this mode has no --dryrun: producing the combined table is the
    entire point of running it, and doing so always requires the source files to exist,
    so file-existence checks are never skipped.

    Args:
        config_path: Path to the config file.
    """
    cfg = get_dict_from_config_file(config_path)
    config: AppConfig = validate_config(cfg, config_path, dryrunQ=False)
    print(f"[SUCCESS] Config validation passed for '{config_path}'.")

    combined_rows: List[Dict[str, Any]] = []
    for block in config.outer_loops:
        combined_rows.extend(_evaluate_tabular_rows(block))
    print(f"[INFO] Extracted {len(combined_rows)} row(s) from {len(config.outer_loops)} source(s).")

    process_cfg = config.process
    column_names = process_cfg.column_names if process_cfg else None
    if column_names:
        # Positional rename: schema validation already guarantees every row has exactly
        # len(column_names) values, since all sources were required to select the same
        # column count.
        combined_rows = [dict(zip(column_names, row.values())) for row in combined_rows]

    if process_cfg and process_cfg.cuts:
        before = len(combined_rows)
        combined_rows = [
            row for row in combined_rows
            if all(_evaluate_cut(expr, row) for expr in process_cfg.cuts)
        ]
        print(f"[INFO] Cuts kept {len(combined_rows)}/{before} row(s).")

    output_path = config.output_file_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = column_names or (list(combined_rows[0].keys()) if combined_rows else [])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"[SUCCESS] Wrote {len(combined_rows)} row(s) to '{output_path}'.")
