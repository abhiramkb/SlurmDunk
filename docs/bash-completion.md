# How bash completion works

Notes on `completions/slurmdunk.bash`: what it completes, how bash finds and loads it, how it was installed on CSC's Roihu, and how it is kept in step with the command line.

---

## 1. What it completes

Press Tab while typing a `slurmdunk` command:

| You type, then Tab | Suggestions |
| :--- | :--- |
| `slurmdunk ` | `generate`, `collect`, config files (`.yaml`, `.yml`, `.json`) and folders |
| `slurmdunk --` | `--help`, `--version` |
| `slurmdunk generate --no` | `--noargcheck`, `--nodupcheck`, `--notimecheck` |
| `slurmdunk generate ` | config files and folders |
| `slurmdunk my.yaml --su` | `--submit`: a bare config path means `generate`, as with the command itself |
| `slurmdunk collect --f` | `--format` |
| `slurmdunk collect --format ` | `sqlite`, `csv`, `both` |
| `slurmdunk collect --output-dir ` | folders only |
| `slurmdunk collect --job-id ` | nothing: a job ID is free text |

Other files, such as `notes.txt`, are never offered where a config path goes, and folders are offered so you can descend into them.

---

## 2. How bash finds it

Bash itself only knows a completion once something has run `complete … slurmdunk`. The script does exactly that on its last line:

```bash
complete -o filenames -F _slurmdunk slurmdunk sd
```

This tells bash to call the function `_slurmdunk` whenever Tab is pressed after `slurmdunk` or its short name `sd`. `-o filenames` makes bash treat the suggestions as file names, so a folder gets a trailing `/` and no space.

The script has to be *sourced* by your shell before that line runs. The **bash-completion** package, which most Linux systems load in interactive shells, does this on demand. The first time you press Tab after a command it has no completion for, it looks for a file named after the command in, in order:

1. `$BASH_COMPLETION_USER_DIR/completions/`, or if that is unset, `${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions/`. This is your personal folder.
2. `bash-completion/completions/` under each directory in `$XDG_DATA_DIRS` (by default `/usr/local/share` and `/usr/share`). These are the system folders.

If it finds a file called `slurmdunk` (or `sd`, for Tab after `sd`), it sources it, and the `complete` line above takes effect. Nothing needs to be added to `~/.bashrc`, and the file is only read when you first use it.

---

## 3. Installing it

With bash-completion present, link the script into your personal folder under each command's name:

```bash
dir="${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions"
mkdir -p "$dir"
ln -s /path/to/SlurmDunk/completions/slurmdunk.bash "$dir/slurmdunk"
ln -s /path/to/SlurmDunk/completions/slurmdunk.bash "$dir/sd"
```

- **Why two links:** bash-completion looks for a file named after the command being completed, so `sd` needs its own link, even though both point to the same script.

- **Why a symlink:** a symlink rather than a copy means a later `git pull` of SlurmDunk updates the completion too.
- **Which variable:** use `$XDG_DATA_HOME` in the command as written, not a hard-coded `~/.local/share`, because some systems set it elsewhere (see section 4).
- **Without bash-completion:** if `type _init_completion` reports "not found" in your shell, bash-completion is not loaded. Source the script directly from `~/.bashrc` instead:

  ```bash
  source /path/to/SlurmDunk/completions/slurmdunk.bash
  ```

**Checking it works** in a new terminal:

```bash
complete -p slurmdunk
```

This should print `complete -o filenames -F _slurmdunk slurmdunk`, either after you have pressed Tab once, or at once if you used `source`.

---

## 4. On Roihu (CSC)

Roihu sets several per-user locations **per CPU architecture**. On the aarch64 GPU login nodes:

| Variable | Value |
| :--- | :--- |
| `XDG_DATA_HOME` | `~/.local/share/aarch64` |
| `PYTHONUSERBASE` | `~/.local/aarch64` (where `pip install --user` puts `slurmdunk`) |

So bash-completion looks in `~/.local/share/aarch64/bash-completion/completions/`, **not** the generic `~/.local/share/bash-completion/completions/`. A link in the generic folder is silently ignored: bash then falls back to its generic `_minimal` completion, and `complete -p slurmdunk` shows `-F _minimal`. That is what happened on the first attempt here.

**Installed for this account** (2026-10-12):

```
~/.local/share/aarch64/bash-completion/completions/slurmdunk
    -> /projappl/lappi/abhiram/SlurmDunk/completions/slurmdunk.bash
~/.local/share/aarch64/bash-completion/completions/sd
    -> /projappl/lappi/abhiram/SlurmDunk/completions/slurmdunk.bash
```

**Verified** by asking bash-completion's own loader to load it, as a Tab press would, and then calling the function:

```bash
bash -ic '_completion_loader slurmdunk; complete -p slurmdunk'
# complete -o filenames -F _slurmdunk slurmdunk
bash -ic '_completion_loader sd; complete -p sd'
# complete -o filenames -F _slurmdunk sd
```

**Other architectures:** if you also work on x86 login nodes, check `echo $XDG_DATA_HOME $PYTHONUSERBASE` there. If they are set per architecture as well, both the `pip install --user -e …` and this symlink need repeating on that architecture.

---

## 5. How the script works

The script is plain bash, deliberately. A Python-based completer (such as `argcomplete`) would start Python and import SlurmDunk, with pydantic, on every Tab, which takes noticeable time on a shared login node. Plain bash answers instantly.

It has three parts.

**Word lists**, one per context, at the top of the file:

```bash
_slurmdunk_commands="generate collect"
_slurmdunk_global_opts="-h --help --version"
_slurmdunk_generate_opts="-h --help --submit --dryrun --noargcheck --notimecheck --nodupcheck"
_slurmdunk_collect_opts="-h --help --job-id --format --output-dir"
_slurmdunk_formats="sqlite csv both"
```

**`_slurmdunk`**, called by bash on every Tab. bash passes the command line as the array `COMP_WORDS`, with `COMP_CWORD` the index of the word being completed. The function fills `COMPREPLY` with the suggestions:

1. **The previous word takes a value.** If it is `--format`, `--output-dir` or `--job-id`, offer that option's values (formats, folders, or nothing) and stop.
2. **Find the subcommand.** It is the first word after `slurmdunk` that isn't an option. If that word is a config path rather than `generate`/`collect`, the subcommand is `generate`: the CLI inserts `generate` in front of a bare config path, so completion does the same.
3. **No subcommand yet:** offer the global options if the word starts with `-`, otherwise the subcommands plus config files.
4. **After a subcommand:** offer that subcommand's options if the word starts with `-`, otherwise config files.

**`_slurmdunk_configs`** adds files ending in `.yaml`, `.yml` or `.json`, plus folders. It uses one `compgen -f -X "!*.ext"` per extension, so it doesn't depend on bash's `extglob` option being enabled.

**Known limitation:** the `--option=value` form is not completed. Bash splits the word at `=`, so `--format=<Tab>` sees `=` as the previous word. Use the space form, `--format <Tab>`.

---

## 6. Keeping it in sync with the command line

The word lists duplicate what `src/slurmdunk/cli.py` defines with `argparse`, so they could drift apart when an option is added or renamed. `tests/test_completion.py` prevents that.

| Test | Checks |
| :--- | :--- |
| `test_word_lists_match_the_cli` | Each word list equals what the real parser (`build_arg_parser()`) defines: the subcommands, the global options, each subcommand's options, and the choices of `--format` |
| `test_every_option_taking_a_value_is_handled` | Every option that takes a value has its own branch in step 1 above, so Tab after it offers values, not config files |
| `test_suggestions` (11 cases) | Runs `_slurmdunk` in bash for typed command lines, in a folder holding configs, a `.txt` file and a subfolder, and compares the suggestions, including the bare-config-path case |

They need no SLURM, and run with the rest of the suite (`python3 -m pytest`). As a check that the sync test bites, renaming `--nodupcheck` in a copy of `cli.py` made `test_word_lists_match_the_cli` fail.

**When you add or rename an option in `cli.py`:**
1. Update the matching word list in `completions/slurmdunk.bash`.
2. If the option takes a value, add a branch for it under `case "$prev" in`.
3. Run `python3 -m pytest tests/test_completion.py`.

A new subcommand also needs its name in `_slurmdunk_commands`, its own `_slurmdunk_<name>_opts` list, and a case in step 2 of `_slurmdunk`.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
| :--- | :--- |
| Tab after `slurmdunk` suggests ordinary files only; `complete -p slurmdunk` shows `-F _minimal` | bash-completion didn't find the file. Check that the link sits in `${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions/slurmdunk` (section 4 on Roihu), then open a new terminal. |
| It works in new terminals but not in one that was already open | That shell pressed Tab after `slurmdunk` before the file existed and cached the fallback. Open a new terminal, or run `complete -r slurmdunk` and press Tab again. |
| `type _init_completion` reports "not found" | bash-completion isn't loaded in this shell. Source the script from `~/.bashrc` (section 3). |
| A newly added option is missing from the suggestions | The word list wasn't updated; `tests/test_completion.py` fails in that case. See section 6. |
