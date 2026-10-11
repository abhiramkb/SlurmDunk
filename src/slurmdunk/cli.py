"""The 'slurmdunk' command: generate/submit SLURM scripts and collect their results."""
import argparse
import sys
from typing import List, Optional

from . import __version__
from .collect import FORMAT_CHOICES, collect_results
from .checks import check_duplicate_submission
from .config_schema import AppConfig, validate_config
from .generate import generate_slurm_script, get_dict_from_config_file
from .submit import submit_slurm_script
from .utils import strip_hyphens

COMMANDS = ("generate", "collect")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the two-tier CLI: a subcommand selects the mode, and each subcommand
    owns its own flags, so e.g. '--job-id' cannot be passed to 'generate' and
    '--submit' cannot be passed to 'collect'.
    """
    # Shared by every subcommand, defined once via a parent parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "config_paths",
        nargs="*",
        default=["config.json"],
        help="Path to the JSON/YAML configuration file(s) (default: config.json)"
    )

    parser = argparse.ArgumentParser(
        prog="slurmdunk",
        description="Generate and submit SLURM scripts from JSON/YAML config files, "
                     "and collect their results."
    )
    parser.add_argument("--version", action="version", version=f"slurmdunk {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", parents=[common],
        help="Generate (and optionally submit) a SLURM script from a config. Default when no subcommand is given."
    )
    generate.add_argument(
        "--submit", action="store_true",
        help="Automatically submit the generated script to SLURM."
    )
    generate.add_argument(
        "--dryrun", action="store_true",
        help="Generate without touching SLURM; also relaxes file-existence checks."
    )
    generate.add_argument(
        "--noargcheck", action="store_true",
        help="Disable checking of whether args passed to the executable are supported."
    )
    generate.add_argument(
        "--notimecheck", action="store_true",
        help="Disable checking the requested time limit against the partition's MaxTime."
    )
    generate.add_argument(
        "--nodupcheck", action="store_true",
        help="Disable skipping configs that already have queued or running jobs generated from them."
    )

    collect = subparsers.add_parser(
        "collect", parents=[common],
        help="Collect one row per run (arguments, status, timing, values matched by the config's "
             "'collect' patterns, JSON output) into results.sqlite and/or results.csv."
    )
    collect.add_argument(
        "--job-id", default=None,
        help="Collect only this job (or one array task, as <arrayjobid>_<task>), into results_<id>.*."
    )
    collect.add_argument(
        "--format", choices=list(FORMAT_CHOICES), default=None,
        help="Output format(s); overrides 'collect.formats' in the config (default: sqlite)."
    )
    collect.add_argument(
        "--output-dir", default=None,
        help="Write the results here instead of the experiment directory."
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    # 'slurmdunk my_config.json ...' (no subcommand) is treated as 'generate'. Only the
    # exact absence of a recognized subcommand triggers this - an explicit subcommand,
    # '-h'/'--help' or '--version' is left untouched.
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version")):
        argv = ["generate"] + argv

    args = build_arg_parser().parse_args(argv)

    if args.command == "collect":
        for config_path in args.config_paths:
            print(f"\n--- Processing: {config_path} ---")
            # File checks relaxed as for --dryrun: input data may be gone by collect time
            config: AppConfig = validate_config(get_dict_from_config_file(config_path), config_path, True)
            collect_results(
                config,
                {strip_hyphens(str(name)) for name in config.argument_names()},
                job_id=args.job_id,
                formats=FORMAT_CHOICES[args.format] if args.format else None,
                output_dir=args.output_dir,
            )

    else:  # generate
        checkargsQ = not args.noargcheck

        for config_path in args.config_paths:
            print(f"\n--- Processing: {config_path} ---")

            # Checked per config, not once up front, so a config listed twice in
            # one '--submit' invocation sees the jobs its first pass submitted.
            if not args.dryrun and not args.nodupcheck and check_duplicate_submission(config_path):
                continue

            generated_script, backup_dir = generate_slurm_script(
                config_path, args.dryrun, checktimeQ=not args.notimecheck
            )

            if generated_script is None:
                print(f"[INFO] Nothing to submit for {config_path}: all tasks already checkpointed.")
                continue

            if args.dryrun:
                print(f"[INFO] Dry-run: not submitting to SLURM for {config_path}.")
            elif args.submit:
                cfg = get_dict_from_config_file(config_path)
                config: AppConfig = validate_config(cfg, config_path, args.dryrun)
                submit_slurm_script(generated_script, config, checkargsQ, backup_dir)
            else:
                print(f"[TIP] Run 'sbatch {generated_script}' to manually submit, or pass --submit next time.")


if __name__ == "__main__":
    main()
