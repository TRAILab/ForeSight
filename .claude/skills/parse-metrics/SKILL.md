---
name: parse-metrics
description: Extracts key training or evaluation metrics from remote ForeSight logs when given a server plus a config stem or job ID.
---

## Inputs
Parse from: `$ARGUMENTS`
- `--server` (required): `dgx` | `apollo` | `narval` | `trillium` | `killarney`
- `--job <id>`: SLURM job ID for DGX or Narval
- `--config <stem>`: config stem such as `auto_apr26_exp001_foo`

Provide at least one of `--job` or `--config`. Prefer `--config` when both are available because the work-dir log is usually more complete.

Use the host and repo mappings from `.claude/CLAUDE.md`.

## Log selection
- Prefer `work_dirs/<stem>/*.log` when `--config` is available.
- For DGX or Narval, fall back to `logs/foresight-<JOB_ID>.log` when `--job` is available.
- If no log exists, report `No logs found yet - job may still be running or work_dir not created.`

## Command
```bash
ssh <host> "cat <log>" | grep -E "NDS|mAP|ade=|epa=|L2|obj_box_col|mAP_normal" | tail -30
```

## Output
Use this table shape and mark missing values as `-`:

| Metric | Value |
|--------|-------|
| L2 | x.xx |
| obj_box_col | x.xx% |
| car_ade | x.xx |
| ped_ade | x.xx |
| car_epa | x.xx |
| ped_epa | x.xx |
| NDS | x.xx |
| mAP | x.xx |
| mAP_normal | x.xx |

Add one status hint:
- If the log exists but metrics are missing: `Training may still be in progress.`
- If the log contains `Error` or `Traceback`: `Job appears to have crashed; inspect the full log.`

## Rules
- Apollo does not use SLURM logs, so `--config` is effectively required there.
- On Trillium and Killarney, SLURM logs are at `/scratch/spapais/ForeSight/logs/` (not home).
- Prefer the most recent metrics near the end of the log, not intermediate training output.
