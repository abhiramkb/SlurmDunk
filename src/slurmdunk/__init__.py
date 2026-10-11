"""SlurmDunk: config-driven parameter sweeps on SLURM.

Turns a JSON/YAML config into a self-contained SLURM submission script (single
run, inner loop, or job array over outer combinations), resumes from per-point
checkpoints, and collects every run's arguments, status and output into
SQLite/CSV.
"""

__version__ = "0.1.0"
