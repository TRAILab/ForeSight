Pull work_dirs from a remote server back to local, excluding large binary files.

## Arguments
Parse from: $ARGUMENTS
- `--server` (default: `all`): `dgx` | `apollo` | `narval` | `all`

## Steps

1. Run `/connect-server --server <server>` to ensure VPN and SSH are active. Stop if it fails.

2. Run rsync (pull direction — remote → local):
```bash
# DGX
rsync -av --exclude='*.pkl' --exclude='*.pth' spapais@192.168.42.200:/raid/home/spapais/ForeSight/work_dirs/ ./work_dirs/

# Apollo
rsync -av --exclude='*.pkl' --exclude='*.pth' spapais@129.97.163.137:/home/spapais/ForeSight/work_dirs/ ./work_dirs/

# Narval
rsync -av --exclude='*.pkl' --exclude='*.pth' spapais@narval.alliancecan.ca:/home/spapais/ForeSight/work_dirs/ ./work_dirs/
```

For `--server all`, run all three in sequence.

3. Report how many files were synced per server.

## Notes
- `.pkl` and `.pth` files are excluded — logs and metric files only.
- Code changes go via git (push here, pull on server) — this skill is for results only.
