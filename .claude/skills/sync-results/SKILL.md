---
name: sync-results
description: Pulls remote `work_dirs` logs and metrics back to the local repo while excluding checkpoints and other large binary artifacts.
---

## Inputs
Parse from: `$ARGUMENTS`
- `--server` (default: `all`): `dgx` | `apollo` | `narval` | `trillium` | `killarney` | `all`

Use the host and repo mappings from `.claude/CLAUDE.md`.

## Command
Pull `work_dirs/` from the remote repo with `rsync`:
```bash
rsync -av --exclude='*.pkl' --exclude='*.pth' trail_dgx:/raid/home/spapais/ForeSight/work_dirs/ ./work_dirs/
rsync -av --exclude='*.pkl' --exclude='*.pth' apollo:/home/spapais/ForeSight/work_dirs/ ./work_dirs/
rsync -av --exclude='*.pkl' --exclude='*.pth' narval:/home/spapais/ForeSight/work_dirs/ ./work_dirs/
rsync -av --exclude='*.pkl' --exclude='*.pth' trillium:/scratch/spapais/ForeSight/work_dirs/ ./work_dirs/
rsync -av --exclude='*.pkl' --exclude='*.pth' killarney:/scratch/spapais/ForeSight/work_dirs/ ./work_dirs/
```

## Rules
- Pull only from remote to local.
- Exclude `.pth` and `.pkl` files.
- Code changes belong in git, not `rsync`.
