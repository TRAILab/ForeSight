Check active and recent SLURM jobs on DGX and/or Narval, mapping job IDs back to config names.

## Arguments
Parse from: $ARGUMENTS
- `--server` (default: `all`): `dgx` | `narval` | `all`

## Steps

### 1. Query SLURM queue
For each target server (DGX uses SLURM; Apollo does NOT — skip Apollo always):

```bash
# DGX
ssh trail_dgx "source ~/.bashrc && squeue -u spapais -o '%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null"

# Narval
ssh narval "squeue -u spapais -o '%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null"
```

### 2. Map job IDs to config stems
For each running/pending job, find the associated work_dir to identify the config:

```bash
# DGX
ssh trail_dgx "ls -dt /raid/home/spapais/ForeSight/work_dirs/*/ 2>/dev/null | head -10"

# Narval
ssh narval "ls -dt /home/spapais/ForeSight/work_dirs/*/ 2>/dev/null | head -10"
```

Cross-reference by modification time against job start time where possible.

### 3. Format output
Present a clean summary per server:

```
=== DGX ===
JOB_ID   STATE     TIME      CONFIG
------   -----     ----      ------
123456   RUNNING   2:14:32   auto_apr26_exp002_lr_warmup
123455   PENDING   0:00:00   auto_apr26_exp003_dropout

=== Narval ===
(no active jobs)
```

### 4. Hint next actions
For each RUNNING job, suggest:
- "Parse metrics so far: `/parse-metrics --server dgx --job <id> --config <stem>`"

For FAILED/CANCELLED jobs not yet logged, suggest:
- "Check crash log: `ssh trail_dgx tail -50 /raid/home/spapais/ForeSight/logs/foresight-<id>.log`"

## Notes
- Apollo uses Docker direct execution (no SLURM) — there is no queue to check.
- If SSH to DGX fails, remind the user to run `/connect-server --server dgx`.
