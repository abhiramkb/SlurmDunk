# Bash completion for the slurmdunk command.
#
# Enable it with bash-completion (loaded on first use):
#   mkdir -p ~/.local/share/bash-completion/completions
#   ln -s /path/to/SlurmDunk/completions/slurmdunk.bash ~/.local/share/bash-completion/completions/slurmdunk
# or source this file from ~/.bashrc.
#
# The word lists below must match src/slurmdunk/cli.py; tests/test_completion.py checks that.

_slurmdunk_commands="generate collect"
_slurmdunk_global_opts="-h --help --version"
_slurmdunk_generate_opts="-h --help --submit --dryrun --noargcheck --notimecheck --nodupcheck"
_slurmdunk_collect_opts="-h --help --job-id --format --output-dir"
_slurmdunk_formats="sqlite csv both"

# Config files (.yaml, .yml, .json) and directories to descend into
_slurmdunk_configs() {
    local cur="$1" IFS=$'\n' ext
    for ext in yaml yml json; do
        COMPREPLY+=($(compgen -f -X "!*.$ext" -- "$cur"))
    done
    COMPREPLY+=($(compgen -d -- "$cur"))
}

_slurmdunk() {
    local cur="${COMP_WORDS[COMP_CWORD]}" prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

    # Values for the options that take one
    case "$prev" in
        --format) COMPREPLY=($(compgen -W "$_slurmdunk_formats" -- "$cur")); return 0 ;;
        --output-dir) local IFS=$'\n'; COMPREPLY=($(compgen -d -- "$cur")); return 0 ;;
        --job-id) return 0 ;;
    esac

    # The subcommand is the first word that isn't an option. A config path there
    # means 'generate', exactly as the command itself treats it.
    local cmd="" i
    for ((i = 1; i < COMP_CWORD; i++)); do
        case "${COMP_WORDS[i]}" in
            generate | collect) cmd="${COMP_WORDS[i]}"; break ;;
            -*) ;;
            *) cmd=generate; break ;;
        esac
    done

    if [[ -z "$cmd" ]]; then
        if [[ "$cur" == -* ]]; then
            COMPREPLY=($(compgen -W "$_slurmdunk_global_opts" -- "$cur"))
        else
            COMPREPLY=($(compgen -W "$_slurmdunk_commands" -- "$cur"))
            _slurmdunk_configs "$cur"
        fi
    elif [[ "$cur" == -* ]]; then
        local opts="$_slurmdunk_generate_opts"
        [[ "$cmd" == collect ]] && opts="$_slurmdunk_collect_opts"
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
    else
        _slurmdunk_configs "$cur"
    fi
    return 0
}

complete -o filenames -F _slurmdunk slurmdunk
