---
name: job-status
description: Checks running and queued jobs on DGX, Apollo, Narval, Trillium, or Killarney when the user asks for remote job status.
---

## Inputs
Parse from: `$ARGUMENTS`
- `--server` (default: `all`): `dgx` | `apollo` | `narval` | `trillium` | `killarney` | `all`

Use the host and VPN rules from `.claude/CLAUDE.md` as needed.

## Commands

### DGX
```bash
ssh trail_dgx "source ~/.bashrc && squeue -u spapais --format='%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null"
```

### Narval
```bash
ssh narval "source ~/.bashrc && squeue -u spapais --format='%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null"
```

### Trillium
```bash
ssh trillium "source ~/.bashrc && squeue -u spapais --format='%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null"
```

### Killarney
```bash
ssh killarney "source ~/.bashrc && squeue -u spapais --format='%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null"
```

### Apollo
```bash
ssh apollo "docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}' 2>/dev/null"
```

## Output
- Print one table per server.
- Include job ID, name, state, runtime, and queue reason when available.
- If no jobs are active, print `No active jobs.`
- For SLURM servers, include `%R` for `PENDING` jobs.
