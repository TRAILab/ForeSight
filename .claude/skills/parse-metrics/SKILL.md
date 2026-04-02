Parse training/evaluation metrics from a remote server job log and format them as a results table.

## Arguments
Parse from: $ARGUMENTS
- `--server` (required): `dgx` | `apollo` | `narval`
- `--job <id>`: SLURM job ID (DGX or Narval only) — used to find SLURM log as fallback
- `--config <stem>`: config stem (e.g. `auto_apr26_exp001_foo`) — used to find work_dir log

At least one of `--job` or `--config` is required. Providing both is fine (work_dir log is preferred).

## Steps

### 1. Find the log file
Try work_dir log first (more complete), fall back to SLURM log:

```bash
# Work dir log (preferred)
LOG=$(ssh <host> "ls -t <remote_repo>/work_dirs/<stem>/*.log 2>/dev/null | head -1")

# SLURM log fallback (DGX/Narval only)
LOG="<remote_repo>/logs/foresight-<JOB_ID>.log"
```

If neither exists, report: "No logs found yet — job may still be running or work_dir not created."

### 2. Grep for metrics
```bash
ssh <host> "cat $LOG" | grep -E "NDS|mAP|ade=|epa=|L2|obj_box_col|mAP_normal" | tail -30
```

### 3. Format output
Parse the grepped lines and present as a clean table:

| Metric | Value |
|--------|-------|
| L2 (avg) | x.xx |
| obj_box_col | x.xx% |
| car_ade | x.xx |
| ped_ade | x.xx |
| car_epa | x.xx |
| ped_epa | x.xx |
| NDS | x.xx |
| mAP | x.xx |
| mAP_normal | x.xx |

If a metric isn't found in the log, mark it as `—`.

### 4. Status hint
If log exists but no metrics found: "Training may still be in progress — metrics appear at end of final epoch."
If job failed (grep finds `Error` or `Traceback`): "Job appears to have crashed — check full log with:
`ssh <host> tail -100 <log>`"

## Notes
- Narval logs WANDB offline; metrics still appear in the log file itself.
- Apollo has no job ID — always use `--config` for Apollo.
