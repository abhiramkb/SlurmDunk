"""Script generation: from a validated config to a self-contained SLURM submission script."""
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Collection, List, Optional, Tuple

import yaml

from . import __version__
from .checks import get_config_tag, validate_partition_time_limit
from .config_schema import (
    AppConfig, ExecutionConfig, ExperimentConfig, SlurmConfig, get_slurm_cpus_per_task, parse_slurm_mem_mb,
    validate_config,
)
from .mode_builders import build_inner_loop_mode, build_job_array_mode, build_single_mode
from .utils import (
    PositionalValueError, bash_dq_escape, format_cli_args, split_positional, tracking_template_to_bash,
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
        config.passed_flags(),
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
    exec_sig_bash = " ".join(
        p for p in exec_sig_components[:3] + [bash_dq_escape(fixed_args_str), bash_dq_escape(fixed_positional_str)] if p
    )

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
        # The same string for embedding inside bash double quotes (quoted values can hold $ or ")
        "fixed_args_bash": bash_dq_escape(fixed_args_str),
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
