---
name: sync-results
description: Pulls remote `work_dirs` logs and metrics back to the local repo while excluding checkpoints and other large binary artifacts.
---

## Inputs
Parse from: `$ARGUMENTS`
- `--server` (default: `all`): `dgx` | `apollo` | `narval` | `all`

Follow the repo remote-access policy in `.claude/CLAUDE.md` before syncing.

## Command
Pull from the remote `work_dirs/` directory with `rsync`:
```bash
rsync -av --exclude='*.pkl' --exclude='*.pth' trail_dgx:/raid/home/spapais/ForeSight/work_dirs/ ./work_dirs/
rsync -av --exclude='*.pkl' --exclude='*.pth' apollo:/home/spapais/ForeSight/work_dirs/ ./work_dirs/
rsync -av --exclude='*.pkl' --exclude='*.pth' narval:/home/spapais/ForeSight/work_dirs/ ./work_dirs/
```

## Rules
- Pull only from remote to local.
- Exclude `.pth` and `.pkl` files.
- Code changes belong in git, not `rsync`.
