# SlurmDunk

SlurmDunk is a lightweight Python based framework for generating Slurm submission scripts (for parameter sweeps), organizing the results of the jobs (including tracking provenance to ensure reproducibility), and parsing the results (on to an SQLite or CSV table).

**Use SlurmDunk if you want to:**

- run a code for various input parameter values, i.e., a parameter scan
  - These parameter values can be read from a table, specified explicitly, or specified as ranges.
- organize the results of your Slurm jobs, and keep track of provenance (i.e., record the git commit of the code you ran to get results, the precise values of the input arguments, environment variables etc.)
- are new to Slurm and want to get started running jobs fast (you should still get educated on Slurm though!)

## "I can write my own Slurm scripts. Why do I need to use SlurmDunk?"

Let's say you are a physics graduate student who wants to compare your theoretical predictions to data. You have a table `experiment_data.csv` which contains the data and a Python code `theory.py` that can calculate you theory predictions. The data file could be something like:

| Energy | Mass | Cross-section |
| ---: | ---: | ---: |
| 1.0 | 0.5 | 12.4 |
| 2.0 | 0.5 | 8.7 |
| 1.0 | 0.6 | 9.0 |
| 2.0 | 0.6 | 9.5 |
| ⋮ | ⋮ | ⋮ |

*and hundreds more such lines...*

1. In this case, you could submit 100s of jobs (perhaps using a Python or bash script to schedule each job), one for each line. But this could take a long time to get scheduled, and more importantly, make your sysadmin *very* angry!

2. Or you could take the more sophisticated approach of submitting a [Slurm Job Array](https://slurm.schedmd.com/job_array.html). But there may be strict limits on the sizes of such arrays in your HPC system (typically, 100, 200 or so). What if you had many more datapoints?

3. Or you may have the misfortune of using [JUPITER](https://www.fz-juelich.de/en/jupiter) (Europe's shiniest, fastest new supercomputer), which doesn't allow different jobs to share the same node — you'd have to waste a whole node to run your small single-threaded code.

**In these situations, SlurmDunk can help you run all your work given a fixed allocation of resources (a whole node, or part of it).**

All you'd have to do is write a config file like:

```yaml
execution:
  interpreter: python3
  executable: <code_directory_path>/theory.py
  multithreading_level: 1           # theory.py is single-threaded

slurm:
  job-name: theory_vs_data
  partition: batch                   # <- your partition
  account: myproject                 # <- your project
  time: "12:00:00"
  ntasks: 1
  cpus-per-task: 64                  # 64 / 1 = 64 runs of theory.py at a time

experiment:
  result_database_path: results
  experiment_name: theory_vs_data
  slrm_output_dir: SLURM_OUTPUT

inner_loop:                          # one run of theory.py per row of the table
  type: tabular_file
  file_path: <path_to_some_dir>/experiment_data.csv
  delimiter: ","
  comment_prefix: Energy             # skips the header row
  arg_names: [energy, mass]          # columns 0 and 1; Cross-section is not passed

collect:
  formats: [sqlite, csv]
  patterns:
    - 'Cross-section: {prediction}'  # picks the number out of a line like "Cross-section: 12.1"
```

Most of the config file may be self-explanatory, while some parts may be less so (like the `experiment` section and the `collect` section, which we will get to very soon). Once you write this YAML config file, you can execute
```console
slurmdunk generate config.yaml --submit
```
This generates and submits a SLURM script that runs the single-threaded `theory.py` on all the data-points, with 64 data-points being processed *concurrently*. For each run of `theory.py`, the arguments `--energy` and `--mass` are passed, with their values being read from `experiment_data.csv` as:
```console
python3 theory.py --energy 1.0 --mass 0.5
python3 theory.py --energy 2.0 --mass 0.5
python3 theory.py --energy 1.0 --mass 0.6
python3 theory.py --energy 2.0 --mass 0.6
...
```

Great! Now you've managed to run the code for all the datapoints. At this point, we can understand that the *inner_loop* section allows us to specify the inputs to (so to speak) loop the executable over. Why is it called an *inner* loop though? The convenient (but not strictly correct, more on this later) answer is that all these points in ```<path_to_some_dir>/experiment_data.csv``` are looped over within a *single* SLURM job. There are however, situations where it might be convenient to split things over multiple jobs. For example, suppose there is a third argument, `--model`. You have 100 different *models*, each indicated by a number from 1 to 100, for which you want to calculte the theory predictions (these could be PDF replicas, for those who know what PDFs are). In this situation, we can generate theory predictions for the experimental data points, and for all theory models, by creating a SLURM job array with a hundred jobs and assigning each job to a theory model. To do this, we add an *outer_loop* section to the config file:
```yaml
outer_loops:
  type: range
  arg_name: model
  start: 1
  step: 1
  stop: 100
```
In this case, we have only a single outer loop, but in principle, we can have multiple loops over separate arguments.

Now that we've seen the basics of how to run jobs, let's see how to collect the results. Result collection is another important feature of SlurmDunk...