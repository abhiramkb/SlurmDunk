"""Submitting a generated script with sbatch."""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .checks import validate_script_args
from .config_schema import AppConfig


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
