Submit a training or evaluation job to a remote server.

## Arguments
Parse from: $ARGUMENTS
- `--server` (required): `dgx` | `apollo` | `narval`
- `--config` (required): config file path relative to repo root (e.g. `projects/configs/sparsedrive_r50_stage2_4gpu.py`)
- `--gpus` (default: infer from server — DGX: 4, Apollo: 8, Narval: 4)
- `--test`: if present, run evaluation instead of training; requires `--ckpt`
- `--ckpt`: checkpoint path (required when `--test`)

## Before submission
1. Run `/connect-server --server <server>` to ensure VPN and SSH are active. Stop if it fails.
2. Confirm the config file exists on the remote:
```bash
ssh <host> "test -f <remote_repo>/<config> && echo exists || echo MISSING"
```
If missing, tell the user to push the branch and pull on the server:
```bash
git push
ssh <host> "cd <remote_repo> && git pull"
```

## Behavior

Derive the inner command:
- Train: `bash ./tools/dist_train.sh <config> <gpus> --deterministic`
- Test:  `bash ./tools/dist_test.sh <config> <ckpt> <gpus> --deterministic --eval bbox`

Then wrap with the server's run script and submit.

### DGX (SLURM)
```bash
ssh trail_dgx "source ~/.bashrc && cd /raid/home/spapais/ForeSight && sbatch --export=ALL scripts/dgx_run.sh <inner_cmd>"
```
- Parse job ID from `Submitted batch job <ID>`
- Print: "Submitted DGX job <ID> — parse results with `/parse-metrics --server dgx --job <ID>`"

### Apollo (Docker, no SLURM)
```bash
ssh apollo "cd /home/spapais/ForeSight && ./scripts/apollo_run.sh <inner_cmd>"
```
- No job ID. Job runs in foreground over SSH.
- Warn: "Apollo runs synchronously — keep SSH session alive or use tmux on Apollo."

### Narval (SLURM)
```bash
ssh narval "source ~/.bashrc && cd /home/spapais/ForeSight && sbatch --export=ALL scripts/narval_run.sh <inner_cmd>"
```
- Parse job ID from `Submitted batch job <ID>`
- Note: Narval runs WANDB in offline mode (set in narval_run.sh).
- Print: "Submitted Narval job <ID>"
