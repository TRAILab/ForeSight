Check running and queued jobs on a remote server.

## Arguments
Parse from: $ARGUMENTS
- `--server` (default: `all`): `dgx` | `apollo` | `narval` | `all`

## Steps

1. Run `/connect-server --server <server>` to ensure VPN and SSH are active. Stop if it fails.

2. Query jobs per server:

### DGX (SLURM)
```bash
ssh trail_dgx "source ~/.bashrc && squeue -u spapais --format='%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null"
```

### Narval (SLURM)
```bash
ssh narval "source ~/.bashrc && squeue -u spapais --format='%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null"
```

### Apollo (Docker — no SLURM)
```bash
ssh apollo "docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}' 2>/dev/null"
```

3. Format output as a clean table per server. Include job ID, name/config, state, and runtime. If no jobs are running, print "No active jobs."

4. For SLURM servers, also note any jobs in PENDING state and why (the `%R` reason field).

## Notes
- SLURM states to highlight: `RUNNING` (active), `PENDING` (queued), `FAILED`/`CANCELLED` (recent failures).
- Job IDs from this output can be passed directly to `/parse-metrics --job <id>` or `/tail-log --job <id>`.
- For `--server all`, run all three checks in sequence and print a combined summary.
