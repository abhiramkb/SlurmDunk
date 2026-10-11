# SlurmDunk

Config-driven parameter sweeps on SLURM: an automated framework for generating and submitting SLURM batch scripts for scientific compute codes, supporting single runs, 1D parameter loops, and multi-dimensional outer/inner parameter scans with resumable checkpointing.

Code conceptualized by me (Abhiram Kaushik) but implemented using the Gemini LLM. Permalink to the chat [here](https://share.gemini.google/WEDBVBqp6mVB)

---

## Overview

`slurmdunk` reads a JSON or YAML configuration file, validates it against a Pydantic schema, expands the parameter space **in Python**, and writes a self-contained Bash submission script.

Parameter expansion happens at generation time: every loop is unrolled into literal argument strings embedded in the generated script, so the script does not read data files at runtime. Work that has already completed is filtered out before the script is written, so a resumed run only submits what is left.

## Installation

From a checkout of this repository:

```bash
python3 -m pip install --user -e /path/to/SlurmDunk
```

- **The command:** this installs `slurmdunk` into your user `bin` directory (`$(python3 -m site --user-base)/bin`, which should be on your `PATH`). pydantic and PyYAML come along as dependencies.
- **Which Python:** install with the Python that should run SlurmDunk. On a cluster that is usually the system `/usr/bin/python3`, not one from a loaded module. The installed command remembers it, so loading other modules later does not matter.
- **Editable install:** with `-e`, changes to the checkout take effect without reinstalling, and generated scripts record the checkout's git commit.
- **Without installing:** `python3 -m slurmdunk …` works too, as long as the package is importable.

## Requirements

**Where `slurmdunk` runs** (usually a login node):
- Python ≥ 3.9. pydantic ≥ 2 and PyYAML are installed with the package.
- The SLURM client commands:
  - `sbatch`, for `--submit`;
  - `scontrol`, for the partition time check (`--notimecheck` skips it);
  - `squeue` with `--me`, for the duplicate-submission check (`--nodupcheck` skips it).

  Where `scontrol` or `squeue` is missing, the check prints a warning and is skipped.

**Where the generated scripts run** (the compute nodes):
- **bash ≥ 5.1.** The concurrent inner loop uses `wait -n -p`.
- **Standard tools:** GNU coreutils (`md5sum`, `date +%N`, `tail`, `wc`) and `sed`.
- **`taskset`** from util-linux. It is optional: it pins concurrent runs to their own cores, and is skipped when absent.
- **Lmod or Environment Modules**, only if `execution.modules` is set.

**Tested on:** CSC's Roihu (Slurm 25.05 and 26.05, bash 5.1.8, Python 3.9.25). Other SLURM clusters that meet the requirements above should work, but have not been tested.

## Command-Line Usage

A subcommand selects the mode; each subcommand has its own flags, so e.g. `--submit` only exists under `generate` and `--format` only under `collect`.

```bash
# Use the default configuration file (config.json)
slurmdunk generate

# One or more explicit config files
slurmdunk generate examples/sweep.yaml
slurmdunk generate config_a.json config_b.yaml

# Generate and submit in one step
slurmdunk generate my_config.json --submit

# Generate without touching SLURM, and skip file-existence checks
slurmdunk generate my_config.json --dryrun

# Collect one row per run into results.sqlite (and/or results.csv)
slurmdunk collect my_config.json
slurmdunk collect my_config.json --job-id 12345678 --format both

# Combine tabular datasets (process_datasets: true configs - see below)
slurmdunk process my_dataset_config.json
```

`generate` is also the default when no subcommand is given at all, so `slurmdunk my_config.json` works too.

| Subcommand | Flag | Description |
| :--- | :--- | :--- |
| `generate` | `--submit` | Submit the generated script with `sbatch`. |
| `generate` | `--dryrun` | Generate the script only; also relaxes file-existence validation. |
| `generate` | `--noargcheck` | Skip probing the executable's `--help` to verify that every configured flag is supported. |
| `generate` | `--notimecheck` | Skip checking the requested `time` against the partition's `MaxTime`. |
| `generate` | `--nodupcheck` | Skip checking the queue for jobs already generated from the same config (see [`slurm`](#slurm)). |
| `collect` | `--job-id <id>` | Collect one job (or one array task, `<arrayjobid>_<task>`) into `results_<id>.*` instead of `results.*`. |
| `collect` | `--format sqlite\|csv\|both` | Output format(s); overrides `collect.formats` in the config. Default `sqlite`. |
| `collect` | `--output-dir <dir>` | Write the results here instead of the experiment directory (e.g. outside a purged `/scratch`). |
| `process` | *(none)* | Always writes the combined table; see [Dataset Processing Mode](#dataset-processing-mode). |

Run `slurmdunk <subcommand> --help` for a subcommand's own flags.

## Execution Modes

The mode is derived from the config; `loopQ` may set it explicitly, otherwise it is inferred from which loop sections are present.

| Mode | Condition | Output | Log prefix |
| :--- | :--- | :--- | :--- |
| **Single run** | `loopQ: false`, or no loop sections | `submit_single_<exp>_<timestamp>.sh` | `[SINGLE_out]` |
| **Inner loop only** | `inner_loop` present, no `outer_loops` | `submit_inner_<exp>_<timestamp>.sh` | `[L1_out]`, `[L2_out]`, … |
| **Job array** | `inner_loop` and `outer_loops` present | `submit_array_<exp>_<timestamp>.sh` | `[A0_L1_out]`, `[A0_L2_out]`, … |

In job array mode the outer combinations form the SLURM array (one task per Cartesian product entry, indexed by `$SLURM_ARRAY_TASK_ID`), and each task iterates the full inner loop sequentially.

---

## Configuration Schema

### Top level

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `execution` | Object | **Yes** | Interpreter, executable, flags, modules, environment variables. |
| `slurm` | Object | **Yes** | Translated into `#SBATCH --<key>=<value>` directives. |
| `loopQ` | Boolean | Optional | Force single-run (`false`) or loop (`true`) mode. Inferred when omitted. |
| `inner_loop` | Object | Conditional | Required in loop mode. The sequential sweep run inside one SLURM task. |
| `outer_loops` | Array | Optional | Turns the run into a SLURM job array over the Cartesian product of these blocks. |
| `experiment` | Object | Optional | Result/checkpoint directory layout and tracking arguments. |
| `args` | Object | Optional | Fixed CLI key-value pairs passed to every invocation. |
| `collect` | Object | Optional | What the `collect` subcommand extracts per run; see [Result Collection](#result-collection). |
| `positional_args` | Array | Optional | Names of arguments passed by position instead of as flags, in command-line order; see [Positional arguments](#positional-arguments). |
| `positional_placement` | String | Optional | `last` (default): positional arguments after all flags. `first`: straight after the executable. |

### `execution`

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `executable` | String | **required** | Path to the program or script. Its parent directory becomes the working directory. |
| `interpreter` | String | `null` | Command that runs the executable, e.g. `julia`, `python3`. Omit for a directly executable file. Also accepted under its legacy name `language`; prefer `interpreter` in new configs. |
| `flags` | Array | `[]` | Interpreter flags placed before the executable, e.g. `["-u"]`. |
| `modules` | Array | `[]` | `module load` calls emitted before execution. |
| `env_vars` | Object | `{}` | Exported before execution. Values may reference SLURM variables, e.g. `"$SLURM_CPUS_PER_TASK"`. |
| `preamble` | String | `null` | Arbitrary, free-form Bash pasted verbatim into the script, after modules/env vars and before execution — e.g. `source`-ing a setup script or building a more complex environment than `modules`/`env_vars` can express. In YAML, use a block scalar (`preamble: \|`) for multi-line content. Processes it starts in the background (e.g. an `nvidia-smi -l 1` monitor) are disowned, so the script does not wait for them: stop them in the `epilogue` via their saved `$!`, or leave them for SLURM to kill when the job ends. |
| `epilogue` | String | `null` | The counterpart of `preamble`: free-form Bash pasted verbatim at the very end of the script, run once every executable invocation has finished (in parallel mode, after all concurrent runs are drained) — e.g. archiving or post-processing results. It runs in the executable's directory and shares the shell with the preamble, so variables set there are still visible. In array mode it runs at the end of **each** array element, not once after the whole array; use a separate job with `--dependency=afterany:<jobid>` for that. It does not run if the job is killed (walltime, `scancel`), and its last command's exit status becomes the job's. |
| `multithreading_level` | Integer | `null` | Cores used by **one** invocation of the executable. When set, the inner loop runs `SLURM_CPUS_PER_TASK / multithreading_level` iterations concurrently. Omit for a strictly sequential inner loop. |
| `max_parallel_tasks` | Integer | `null` | Hard ceiling on concurrent runs, regardless of how many cores are free — mainly to stay within `--mem`. |

### `slurm`

Every key is emitted verbatim as `#SBATCH --<key>=<value>`, so any SBATCH option is accepted (`account`, `mem`, `cpus-per-task`, `ntasks`, …). `partition` and `time` are required; `job-name` defaults to `slurmdunk`, `nodes` to `1`, `output` to `slurm_%j.log`.

`max_concurrent_tasks` is consumed by SlurmDunk rather than emitted: it becomes the `%N` throttle on the array directive (e.g. `#SBATCH --array=0-7%2`).

`num_array_jobs` is likewise consumed rather than emitted, and sets how many array elements the outer loop is spread over — see [Packing the outer loop](#packing-the-outer-loop).

Before a script is written, the requested `time` is checked against the partition's `MaxTime` (via `scontrol`) in the same spirit as the executable argument check: exceeding it, or naming a partition that does not exist, is an error rather than something `sbatch` rejects later. The check is skipped silently where `scontrol` is unavailable, and `--notimecheck` disables it.

Every generated script also carries `#SBATCH --comment="slurmdunk-config:<absolute config path>"`, which tags its jobs with the config they came from (visible in `squeue -o %k` and `sacct`). Before generating, `generate` looks up your queued and running jobs (`squeue --me`) for that tag. Checkpointing only skips work that has finished, so if any are found the config is skipped with an error listing their job IDs, and the remaining configs in the same command carry on. Wait for those jobs to finish, `scancel` them, or pass `--nodupcheck` to submit anyway. The check runs at generation rather than only with `--submit`, since the script is often submitted by hand later. It is skipped under `--dryrun`, prints a warning and continues where `squeue` is unavailable, and skips the config if the queue cannot be read. Jobs tagged `orchestrator-config:` by Code-Orchestrator, the framework's earlier version, are recognised too; jobs submitted before either tag existed are not detected. `comment` is therefore reserved: a value set in the `slurm` block is replaced, with a warning.

### `inner_loop`

Two forms, selected by `type`. A dict with no `type` is treated as `tabular_file` for backward compatibility.

**`tabular_file`** — one run per row of a data file:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file_path` | Path | `null` | Data file to read. |
| `delimiter` | String | `" "` | Column separator; whitespace splitting when a single space. |
| `comment_prefix` | String or Array | `"#"` | Lines starting with this are skipped. Give a list, e.g. `["#", "*", "!"]`, to skip lines starting with any of several prefixes. `""` (or `[]`) disables comment skipping entirely. Each line is stripped of leading/trailing whitespace *before* this check, so an indented header line is matched by its first non-space character, not by the indentation. |
| `skip_blank_lines` | Boolean | `true` | Skip empty lines. |
| `arg_names` | Array | `null` | Shorthand: maps columns positionally to argument names. |
| `args` | Array | `[]` | Explicit column specs; use instead of `arg_names` when you need `template` or `transform`. |

**`explicit`** — one run per listed value:

| Field | Type | Description |
| :--- | :--- | :--- |
| `arg_name` | String | Argument name. |
| `values` | Array | Values to sweep. |

### `outer_loops`

An array of blocks; the run expands over their Cartesian product. Each block **must** declare `type`:

- **`explicit`** — `arg_name` plus `values`.
- **`range`** — `arg_name`, `start`, `stop`, `step` (default `1.0`); `stop` is exclusive.
- **`tabular_file`** — `file_path` plus `args` column specs (also `delimiter`, `comment_prefix`, `skip_blank_lines`).

### Column specs (`args` entries)

| Field | Type | Description |
| :--- | :--- | :--- |
| `arg_name` | String | Argument name. Bare names gain `--` (or `-` if single character); names already starting with `-` are passed through unchanged. |
| `source` | String | Where the value comes from: `column` (default) reads a column of the row, `row_index` uses the row's own position. |
| `column` | Integer | Zero-based column index. Required for `source: column`, and rejected for `source: row_index`. |
| `index_offset` | Integer | Only for `source: row_index`. Added to the index, e.g. `1` for one-based numbering. Default `0`. |
| `transform` | String | Optional math expression applied to the value, e.g. `"10^x"`, `"log10(x)"`, `"sqrt(x)"`. `x` and `val` both refer to the raw value; `^` means exponentiation. Standard `math` functions plus SciPy special functions (Bessel, gamma, …) are available when SciPy is installed. |
| `template` | String | Optional string template applied after `transform`, e.g. `"/data/bks/{val}.dat"`. |

#### Numbering rows instead of reading them

Some data files carry no index column even though each row corresponds to a numbered file. `source: row_index` supplies that number:

```yaml
inner_loop:
  type: tabular_file
  file_path: /projappl/lappi/abhiram/bayesian_alldata/LO_MVe/posteriorsamples.dat
  delimiter: " "
  comment_prefix: "#"
  args:
    - arg_name: dipole_path
      source: row_index
      template: /projappl/lappi/abhiram/bayesian_alldata/LO_MVe/bks/{val}.dat
    - arg_name: Csq
      column: 2
```

The index counts **retained rows**, so comment and blank lines do not advance it: in a file with a four-line header, the first data row is index `0` and maps to `bks/0.dat`.

The value stays an integer until `template` is applied, so a format spec works for zero-padded names — `"bks/{val:03d}.dat"` gives `bks/000.dat`. Use `index_offset: 1` for one-based files.

> Because the index counts retained rows, changing `comment_prefix` or `skip_blank_lines` renumbers them. That changes the generated arguments and therefore the checkpoint hashes, so completed work under the old settings will not be recognised.

### `experiment`

| Field | Type | Description |
| :--- | :--- | :--- |
| `result_database_path` | String | Root directory for results. |
| `experiment_name` | String | Subdirectory under the root; also used in generated script names. |
| `slrm_output_dir` | String | Subdirectory for SLURM logs. |
| `tracking_args` | Object | Extra CLI arguments injected per run, supporting the placeholders `{result_database_path}`, `{experiment_name}`, `{slrm_output_dir}`, `{save_dir}` and `{__indicator}`. |

This yields the layout:

```text
<result_database_path>/<experiment_name>/
├── <slrm_output_dir>/       # SLURM logs
├── checkpoints/             # completion markers
├── <job_id>_<indicator>/    # per-run output directories ($SAVE_DIR)
└── results.sqlite           # written by 'collect' (results.csv with --format csv/both)
```

Without an `experiment` block, checkpoints go to `./checkpoints` relative to the submission directory and no result directories are created.

### Positional arguments

Every argument is passed as a flag (`--name value`, or `-n value` for a single letter) unless it is named in `positional_args`. That list makes the named arguments positional and fixes their order on the command line. Each argument is still defined where it normally is — in `args`, a loop or `tracking_args` — so its value can come from anywhere, and the list alone decides the order.

For a program read as `prog <infile> <resfile> <rtol> <repeats>`:

```yaml
args:
  repeats: 5                    # fixed

outer_loops:
  - type: explicit
    arg_name: infile            # outer loop: one array job per file
    values: [a.dat, b.dat]

inner_loop:
  type: explicit
  arg_name: rtol                # inner loop: one run per value
  values: ["1e-6", "1e-8"]

experiment:
  result_database_path: /scratch/me/db
  experiment_name: timing
  slrm_output_dir: SLURM_OUTPUT
  tracking_args:
    resfile: "{save_dir}_{__indicator}/res.dat"   # per-run output path

positional_args: [infile, resfile, rtol, repeats]
```

Array job 4242 then runs:

```text
prog a.dat /scratch/me/db/timing/4242_A0_L1/res.dat 1e-6 5
prog a.dat /scratch/me/db/timing/4242_A0_L2/res.dat 1e-8 5
prog b.dat /scratch/me/db/timing/4242_A1_L1/res.dat 1e-6 5
prog b.dat /scratch/me/db/timing/4242_A1_L2/res.dat 1e-8 5
```

**Placement and flags:**
- **Mixing with flags:** arguments not in the list stay flags, and with the default `positional_placement: last` the positional arguments follow all of them, e.g. `prog --verbose a.dat …`. That is the order strict `getopt` programs need.
- **Subcommand-style programs:** `positional_placement: first` puts the positional arguments straight after the executable instead.
- **Awkward values:** values are quoted when the command is built, so spaces, quotes and `$` reach the program unchanged.

**Rules:**
- **One definition each:** every listed name must be defined exactly once.
- **No dashes:** list and define names without leading dashes, since a positional argument has no flag.
- **Values must stand alone:** a value may not be `true`/`false`, null or empty, because it would shift every later argument. This is checked for fixed values when the config loads, and for loop values when the script is generated; the error names the row.

**Checkpoints and logs:**
- **Hashing:** fixed and loop positional values are part of each point's checkpoint hash, so points differing only in a positional value are tracked separately. Tracking-arg positional values are left out, as tracking flags are. Configs without `positional_args` hash exactly as before.
- **Logging:** positional values are logged as `[<indicator>_args] <name>: <value>` like flags, so `collect` gives each one a column, and `json.path` can use them as placeholders, e.g. `json: "{resfile}"`.

**The `--help` check** (with `--submit`):
- **Flags only:** only flags are looked up in the help text.
- **Positional count:** the number of positional arguments is compared with the program's usage line (`usage: prog [-h] [--x X] infile outfile`, as argparse and ArgParse.jl print it). Usage formats vary, so a mismatch is a warning, not an error.
- **No flags, no check:** a config with no flags at all never runs `--help`. That suits programs that only read `ARGS` and would take `--help` as their first input. If such a program also takes flags, use `--noargcheck`.

---

## Examples

### Case 1: Job array over posterior samples

Outer loop reads dipole fit parameters from a table (one array task per row); each task sweeps every kinematic point in the inner file.

```json
{
  "slurm": {
    "account": "lappi",
    "job-name": "dip_HERA",
    "partition": "small",
    "nodes": 1,
    "ntasks": 1,
    "time": "1-12:59:00",
    "cpus-per-task": 16,
    "mem": "40G",
    "output": "slurm_%j.log",
    "max_concurrent_tasks": 20
  },
  "execution": {
    "interpreter": "julia",
    "executable": "/projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_L.jl",
    "modules": ["openmpi", "julia/1.11.9", "julia-mpi"],
    "env_vars": {
      "JULIA_NUM_THREADS": "$SLURM_CPUS_PER_TASK"
    }
  },
  "experiment": {
    "result_database_path": "/scratch/lappi/abhiram/dip_database",
    "experiment_name": "allbks_mvgam_balsd",
    "slrm_output_dir": "SLURM_OUTPUT",
    "tracking_args": {
      "save_dir": "{save_dir}_{__indicator}",
      "json": "{save_dir}_{__indicator}/result.json"
    }
  },
  "inner_loop": {
    "file_path": "/projappl/lappi/abhiram/trip/HERA_kinematic_points_smallx.txt",
    "delimiter": " ",
    "arg_names": ["--Q", "beta", "--x"]
  },
  "outer_loops": [
    {
      "type": "tabular_file",
      "file_path": "/projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/posteriorsamples.dat",
      "delimiter": ",",
      "comment_prefix": "#",
      "skip_blank_lines": true,
      "args": [
        {
          "arg_name": "dipole_path",
          "column": 0,
          "template": "/projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/bks/{val}.dat"
        },
        {
          "arg_name": "Csq",
          "column": 3,
          "transform": "10^x"
        }
      ]
    }
  ],
  "args": {
    "xmax": 160.0,
    "neval": 4e8
  }
}
```

With 100 posterior samples and 51 kinematic points this emits `#SBATCH --array=0-99%20`, each task running 51 sequential inner evaluations.

### Case 2: Inner loop only

```json
{
  "slurm": {
    "account": "lappi",
    "job-name": "dip_median",
    "partition": "small",
    "nodes": 1,
    "time": "23:59:00",
    "cpus-per-task": 16,
    "output": "slurm_%j.log"
  },
  "execution": {
    "interpreter": "julia",
    "executable": "/projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_L.jl",
    "modules": ["openmpi", "julia/1.11.9"],
    "env_vars": {
      "JULIA_NUM_THREADS": "$SLURM_CPUS_PER_TASK"
    }
  },
  "experiment": {
    "result_database_path": "/scratch/lappi/abhiram/dip_database",
    "experiment_name": "median_mv_balsd",
    "slrm_output_dir": "SLURM_OUTPUT"
  },
  "inner_loop": {
    "file_path": "/projappl/lappi/abhiram/trip/HERA_kinematic_points_smallx.txt",
    "delimiter": " ",
    "arg_names": ["--Q", "beta", "--x"]
  },
  "args": {
    "dipole_path": "/projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/median_bk.dat",
    "Csq": 915,
    "xmax": 160.0
  }
}
```

An explicit sweep works the same way and needs no data file:

```json
{
  "inner_loop": {
    "type": "explicit",
    "arg_name": "Csq",
    "values": [100, 315, 915]
  }
}
```

### Case 3: Single run

```json
{
  "loopQ": false,
  "slurm": {
    "account": "lappi",
    "job-name": "single_dip",
    "partition": "test",
    "nodes": 1,
    "ntasks": 1,
    "time": "00:15:00",
    "cpus-per-task": 16,
    "output": "slurm_%j.log"
  },
  "execution": {
    "interpreter": "julia",
    "executable": "/projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_T.jl",
    "modules": ["openmpi", "julia/1.11.9", "julia-mpi"],
    "env_vars": {
      "JULIA_NUM_THREADS": "$SLURM_CPUS_PER_TASK"
    }
  },
  "experiment": {
    "result_database_path": "/projappl/lappi/abhiram/dip_database",
    "experiment_name": "mvgam_balsd",
    "slrm_output_dir": "SLURM_OUTPUT"
  },
  "args": {
    "dipole_path": "/projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/median_bk.dat",
    "Csq": 915,
    "Q": 1.5811388300841898,
    "beta": 0.18,
    "x": 0.0009,
    "xmax": 160.0,
    "neval": 1e8
  }
}
```

---

## Packing the Outer Loop

By default each outer-loop combination gets its own SLURM array element. `slurm.num_array_jobs` spreads them over a fixed number of elements instead, packing several combinations into each:

```json
"slurm": { "num_array_jobs": 80 }
```

With 100 pending combinations that yields `#SBATCH --array=0-79`: the first 20 elements process 2 combinations each, the remaining 60 process 1. Group sizes always differ by at most one. Omit the key to keep one element per combination, which reproduces the historical sparse `--array=0-2,5,8` behaviour exactly.

This is what makes a sweep larger than the cluster's `MaxArraySize` submittable at all, and it helps where an account's job-submission limit is tight.

Two things to keep in mind:

- **`--time` applies to every element**, so it must cover the *largest* group — that many combinations running back to back. SlurmDunk prints the largest group size at generation time; sizing `time` is your responsibility, and the partition check will catch an over-limit result.
- Packing is computed over the **pending** combinations, so a resumed run repacks whatever is left rather than reserving elements for finished work.

Checkpointing is unaffected. Run indicators key on the outer combination index (`A<combo>_L<line>`), not on the array element, so markers and per-run output directories keep the same names whatever `num_array_jobs` is set to — you can change it mid-campaign and completed work is still recognised.

---

## Parallel Inner Loop

By default the inner loop runs one point at a time. Setting `execution.multithreading_level` to the number of cores a single invocation uses lets SlurmDunk fill the rest of the allocation:

```json
"slurm": { "cpus-per-task": 16 },
"execution": {
  "multithreading_level": 4,
  "env_vars": { "JULIA_NUM_THREADS": "$MULTITHREADING_LEVEL" }
}
```

This runs `16 / 4 = 4` inner points at once. The count is computed at run time from `SLURM_CPUS_PER_TASK`, so it adapts if the allocation changes; it falls back to `nproc`, is capped by `max_parallel_tasks`, and can never drop below 1. Applies to both inner-loop and job-array modes (in array mode it fills the cores *within* each array task).

**Set your thread count yourself.** SlurmDunk exports `MULTITHREADING_LEVEL` for configs to reference but never rewrites `env_vars`. Leaving a thread variable at `$SLURM_CPUS_PER_TASK` while running N ways concurrently requests N times the cores and is *slower* than running sequentially — the generator prints a warning if it spots this, but does not block.

Each run is pinned to its own disjoint slice of the job's CPU mask via `taskset`, so per-process affinity settings such as `JULIA_EXCLUSIVE` bind within a slot instead of every run crowding onto the same cores.

### Output and interruption

Concurrent output stays readable: each run's output is buffered and then emitted as one **contiguous, untorn block**, preserving the usual `[<indicator>_out]` / `[<indicator>_err]` tagging in a single file. Only the parent shell writes to the log, so no locking is involved.

Blocks appear in **completion order**, not `L1, L2, L3` order. This is deliberate — a finished run reaches the log as soon as it completes, so an interrupted job (walltime, `scancel`, crash) still contains every run that finished. Runs still in flight are flushed under a `[PARTIAL]` banner, and while a run is executing its live output can be tailed at `<slrm_output_dir>/.partial/<job_id>/<indicator>.log`.

Checkpointing is unaffected: every concurrent run has a distinct indicator and marker file, and the hashes are identical to those the sequential path produces, so existing `checkpoints/` directories stay valid.

---

## Checkpointing and Resuming

Every individual run is identified by an MD5 hash of its full invocation — interpreter, flags, executable, fixed arguments and loop arguments — so a checkpoint is only reused when the run would be genuinely identical. On success the script touches a marker file:

| Mode | Marker |
| :--- | :--- |
| Single run | `SINGLE_<hash>.done` |
| Inner loop | `L<line>_<hash>.done` |
| Job array | `A<task>_L<line>_<hash>.done` |

`<line>` is the position in the **full** inner loop, so indices remain stable no matter how much of the sweep has already finished.

Completion is checked twice, for different reasons:

1. **At generation time (Python).** Finished work is excluded before the script is written. An inner-loop script contains only the outstanding points; a job array emits a sparse range covering only outer tasks with work left, e.g. `#SBATCH --array=0,5,10,75-99%4`. If nothing is outstanding no script is produced and the run reports `Nothing to submit`.
2. **At run time (Bash).** The script re-checks each marker before executing. This covers a script that is submitted twice or requeued by SLURM after a node failure or time limit.

To resume an interrupted campaign, simply re-run SlurmDunk on the same config: completed points are skipped and only the remainder is submitted.

Deleting a marker forces the corresponding run to repeat; deleting the `checkpoints/` directory restarts the whole campaign.

---

## Dataset Processing Mode

`process_datasets: true` marks a config for the `process` subcommand rather than `generate`: it reads one or more tabular data files, extracts and renames chosen columns, concatenates them row-wise into one table, optionally filters rows, and writes the result — no `execution`, `slurm`, `inner_loop`, checkpointing or submission involved. Run it with `slurmdunk process my_config.json`; there is no `--dryrun` for this subcommand, since writing the combined table is the entire point of running it. Passing such a config to `generate` (or the bare, subcommand-less invocation, which defaults to `generate`) is a clear error telling you to use `process` instead.

| Field | Type | Description |
| :--- | :--- | :--- |
| `process_datasets` | Boolean | Enables this mode. `execution`/`slurm`/`experiment` are not required, and a warning is printed if any are supplied anyway, since they're ignored. |
| `output_file_path` | Path | **Required.** Where the combined table is written, as CSV. |
| `outer_loops` | Array | **Required, at least one entry.** Every entry must be `type: tabular_file` (the same schema used in sweep mode — `file_path`, `args`, `delimiter`, `comment_prefix`, `skip_blank_lines`). Each entry contributes its extracted rows to the combined table. All entries must select the same number of columns, since their rows stack into one table. |
| `process.column_names` | Array | Renames the extracted columns positionally, e.g. the 1st column selected by every source becomes this list's 1st name. Its length must match the column count. Omit to keep each source's own `arg_name`s — only sensible when every source already uses the same names. |
| `process.cuts` | String or Array | One or more boolean expressions; a row is kept only if every cut is true. Evaluated with each of the row's (renamed) columns bound by name, plus the same math functions available to `transform` (`sqrt`, `log10`, SciPy special functions, …). Numeric-looking values are compared numerically; anything else compares as a string. |

```yaml
process_datasets: true
output_file_path: combined_posteriors.csv

process:
  column_names: [Qs0sq, Csq]
  cuts:
    - "Qs0sq < 0.1"

outer_loops:
  - type: tabular_file
    file_path: /projappl/lappi/abhiram/bayesian_alldata/LO_MVe/posteriorsamples.dat
    delimiter: " "
    args:
      - {arg_name: Q, column: 0}
      - {arg_name: C2, column: 2}
  - type: tabular_file
    file_path: /projappl/lappi/abhiram/bayesian_alldata/KCBK_pd_MVgamma/posteriorsamples.dat
    delimiter: ","
    args:
      - {arg_name: Q, column: 1}
      - {arg_name: C2, column: 3, transform: "10**x"}   # this file stores log10(C^2)
```

Each source can use its own `arg_name`s, column order, delimiter and even a `transform` to normalize units before the tables are stacked — only the final column *count* and the shared `column_names` need to agree.

---

## Result Collection

`slurmdunk collect my_config.json` reads the SLURM logs under `<result_database_path>/<experiment_name>/<slrm_output_dir>` and writes **one row per run** to `results.sqlite` (table `runs`) and/or `results.csv` in the experiment directory. It works for any executable, because the generated scripts label every line of a run's output with that run (see [Log Output & Stream Tagging](#log-output--stream-tagging)).

Every row always carries, with no configuration:

| Columns | Source |
| :--- | :--- |
| `job_id` | `<jobid>_<indicator>`, e.g. `2015127_A89_L51`; the table's primary key. |
| `indicator`, `array_task_id`, `log_file` | Which run, array task and log file the row came from. |
| `status`, `exit_code` | `completed` (exit 0), `failed` (non-zero exit) or `interrupted` (started but never finished: walltime, cancellation, node failure). |
| `job_duration` | Seconds, from the script's own timing line. |
| one column per argument | Every argument passed to the executable, whether spelled `-x` or `--beta`, is logged as a `[<indicator>_args] <name>: <value>` line and becomes a column, e.g. `x`, `beta`, `save_dir`. Numbers are stored as numbers; a switch printed without a value is stored as true. An argument named like one of the columns above becomes `arg.<name>`. |

Checkpoint-skipped runs produce no row: a skipped run's row comes from the log of the job that actually ran it. The output is rebuilt on every collect, so re-running after more jobs finish simply picks them up.

### Extracting values from the output: patterns

A pattern is an output line as your code prints it, with each value replaced by a **named placeholder**. The placeholder's name becomes the column:

```yaml
collect:
  formats: [sqlite, csv]                        # default [sqlite]; --format overrides
  patterns:
    - 'Time for integration: {t_int} s'         # short form: just the pattern
    - 'Result of code = [{result},{error}]'
    - pattern: 'Final results: {result} +/- {error}'
      stream: err                               # out | err | any (default any)
      match: last                               # first | last | all (default last)
```

For example, the VEGAS line `[INFO] (vegasflow.monte_carlo)  > Final results: -0.520629 +/- 0.000366309` printed on stderr gives `result = -0.520629`, `error = 0.000366309` with the third pattern above.

| Placeholder | Matches | Stored as |
| :--- | :--- | :--- |
| `{x}` | a number: `10`, `-0.52`, `1.2e-5`, `1.2D-05`, `nan`, `inf` | int if written as an integer, otherwise float |
| `{x:float}` | a number | always float |
| `{x:int}` | an integer | int |
| `{x:str}` | one word without spaces, e.g. a path or a hash (keeps leading zeros) | text |
| `{x:text}` | any text, spaces included | text |

- **No escaping:** everything outside the placeholders is matched literally, so `[`, `(`, `+/-` and `.` are written as they appear. Write `{{` and `}}` for literal braces.
- **Anywhere in the line:** a pattern may match anywhere in a line, so prefixes such as `[INFO] (vegasflow.monte_carlo) >` can be left out.
- **Flexible spacing:** any run of spaces in a pattern matches any run of spaces or tabs, so aligned columns still match. Leading and trailing spaces are ignored.
- **Whole numbers only:** a number is matched whole, so `{n:int}` does not pick the `5` out of `10.5`.
- **Where text placeholders stop:** `str` and `text` stop at the next literal in the pattern, or run to the end of the line when they come last.
- **Repeated matches:** `match` decides what happens when a pattern matches several lines of one run. `last` suits output that converges over iterations; `all` stores every match as a JSON list in one cell.
- **Choosing a stream:** `stream` limits a pattern to the run's stdout (`out`) or stderr (`err`).
- **Raw regex:** for a line a pattern cannot describe, give `regex:` instead of `pattern:`. Each named group (`(?P<name>...)`) becomes a column.

Patterns are checked when the config is loaded, also by `generate`. Each of these is an error that names the offending pattern:
- a pattern without a placeholder;
- an unknown type or an unbalanced brace;
- a placeholder name used twice;
- a placeholder name that reuses an argument name or a built-in column.

At the end of a collect, the number of runs each pattern matched is printed, which shows at once if a pattern never matches.

### JSON output

By default each run's JSON is looked for at `{exp_dir}/{job_id}_{indicator}/result*.json`, which is where the conventional `{save_dir}_{__indicator}` tracking argument puts it. Point elsewhere with `json.path`:

```yaml
collect:
  json:
    path: "{json}"      # or the short form   json: "{json}"
```

- **Placeholders:** the path may use any argument the run printed (`{json}`, `{save_dir}`, …), plus `{job_id}` (the SLURM job ID, as `$JOB_ID` in the script), `{indicator}` and `{exp_dir}`.
- **Wildcards:** `*` and `?` pick the first match in sorted order.
- **Turning it off:** `json: null` disables JSON collection.

**Column names for JSON values:**
- **Leaf names:** nested keys are stored under their leaf name, so `{"metrics": {"result": …}}` becomes the column `result`.
- **Merging:** a leaf equal in name and value to an existing column is merged into it. For example, `parameters.Q` merges into the `Q` argument.
- **Dotted paths:** a leaf keeps its full path (e.g. `metrics.x`) when its name is taken by a different value, when the name occurs twice in the document, or when it matches a pattern or argument name. A top-level key that still clashes gets a `json.` prefix.
- **Other types:** lists and objects are stored as JSON text, and booleans as 1/0.

**Warnings:** a missing file leaves that run's JSON columns empty. A count of missing files is reported only when `json.path` was set explicitly, while unreadable files are always reported.

---

## Log Output & Stream Tagging

Generated scripts print a run legend, then tag every stream line with its run indicator and report nanosecond-precision timing:

```text
======================= SLURM ARRAY RUN LEGEND =======================
Array Task ID: 0
Working Directory: /projappl/lappi/abhiram/NLO_Diffraction_dip-dip
Exec Path: /projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_L.jl
Fixed Args: --xmax 160.0 --neval 4e8
Outer Loop Combo: dipole_path=/data/bks/sample0.dat, Csq=915
Outer Flags: --dipole_path /data/bks/sample0.dat --Csq 915
Inner Args: --Q, --x, beta
Executing Subtasks: 40 / 100 Outer Runs
======================================================================
[A0_L1_args] Q: 1.0
[A0_L1_args] x: 0.001
[A0_L1_out] Job started at: Fri Aug  7 14:00:00 EEST 2026
[A0_L1_out] Job finished at: Fri Aug  7 14:00:02 EEST 2026
[A0_L1_out] Job duration: 2.104 seconds
[CHECKPOINT] Skipping A0_L2 - already completed.
[A0_L3_err] Warning: Convergence threshold near limit.
[A0_L3_out] Job duration: 1.980 seconds
```

---

## Running the tests

```bash
python3 -m pip install --user -e "/path/to/SlurmDunk[test]"   # adds pytest
cd /path/to/SlurmDunk && python3 -m pytest
```

- **No SLURM needed:** the tests generate scripts with `--dryrun`, run them under bash with a stub program, and replace `squeue` with a stub where they need it.
- **bash ≥ 5.1:** tests that run generated scripts are skipped on older versions.
- **Checkpoint names are pinned:** `tests/test_checkpoint_hashes.py` fixes the names of checkpoint files. If it fails, the change would make existing campaigns start again from scratch, so it needs a migration path rather than new expected values.

---

## Notes on Removed Behaviour

Earlier versions allowed an outer loop block to select a different inner-loop data file per value via an `inner_files` mapping. Inner loops are now evaluated once and shared across all outer combinations, so `inner_files` is rejected with an explicit error rather than silently ignored. Use a separate config per inner file if they need to differ.

SlurmDunk grew out of an earlier framework, Code-Orchestrator. That framework's NLO Diffraction specific `collectold` collector is not part of SlurmDunk; `collect` gathers the same data generically (see [Result Collection](#result-collection)). Checkpoint names are unchanged, and the duplicate-submission check still recognises Code-Orchestrator's job tags, so campaigns started with Code-Orchestrator resume under SlurmDunk.

---

## License

MIT; see [LICENSE](LICENSE).
