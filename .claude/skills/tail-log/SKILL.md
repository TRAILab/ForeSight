Stream a live or recent job log from a remote server.

## Arguments
Parse from: $ARGUMENTS
- `--server` (required): `dgx` | `apollo` | `narval`
- `--job <id>`: SLURM job ID (DGX or Narval) — used to find SLURM log
- `--config <stem>`: config stem (e.g. `auto_apr26_exp001_foo`) — used to find work_dir log
- `--lines <n>` (default: 50): number of tail lines to show before following

At least one of `--job` or `--config` is required.

## Steps

1. Run `/connect-server --server <server>` to ensure VPN and SSH are active. Stop if it fails.

2. Resolve the log file path (prefer work_dir log; fall back to SLURM log):
```bash
# Work dir log (preferred — more complete)
LOG=$(ssh <host> "ls -t <remote_repo>/work_dirs/<stem>/*.log 2>/dev/null | head -1")

# SLURM log fallback (DGX/Narval only)
# DGX:    logs/foresight-<JOB_ID>.log
# Narval: logs/foresight-<JOB_ID>.log
LOG="<remote_repo>/logs/foresight-<JOB_ID>.log"
```

3. If no log file is found, print a clear error and stop.

4. Stream the log:
```bash
ssh <host> "tail -n <lines> -f <LOG>"
```

## Server reference
| Server | ssh_host    | remote_repo                        |
|--------|-------------|------------------------------------|
| DGX    | trail_dgx   | /raid/home/spapais/ForeSight       |
| Apollo | apollo      | /home/spapais/ForeSight            |
| Narval | narval      | /home/spapais/ForeSight            |

## Notes
- `tail -f` streams until the user interrupts (Ctrl-C). Let the user know.
- Apollo has no SLURM logs — only work_dir logs are available; `--job` is not applicable.
- If the job has already finished, `tail -f` will still show the full log and then exit naturally once EOF is reached.
