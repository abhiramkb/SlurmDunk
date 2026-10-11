"""Checks before a script is generated or submitted: the partition's time limit, jobs already
queued from the same config, and whether the executable accepts the configured flags."""
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config_schema import AppConfig, parse_slurm_time_seconds
from .mode_builders import build_exec_command


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


def format_seconds_as_slurm_time(total_seconds: int) -> str:
    """Renders seconds as SLURM's D-HH:MM:SS, dropping the day part when zero."""
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    stamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{stamp}" if days else stamp


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
    flags_to_check = config.passed_flags()
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
