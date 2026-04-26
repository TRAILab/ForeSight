---
name: submit-job
description: Submits a ForeSight training or evaluation run to DGX, Apollo, Narval, Trillium, or Killarney when given a server and config path.
---

## Inputs
Parse from: `$ARGUMENTS`
- `--server` (required): `dgx` | `apollo` | `narval` | `trillium` | `killarney`
- `--config` (required): repo-relative config path such as `projects/configs/sparsedrive_r50_stage2_4gpu.py`
- `--gpus`: default by server (`dgx=4`, `apollo=8`, `narval=4`, `trillium=4`, `killarney=4`)
- `--test`: run evaluation instead of training
- `--ckpt`: checkpoint path required with `--test`

Use the host, repo, and VPN rules from `.claude/CLAUDE.md`.

## Preflight
Verify the config exists on the remote host:
```bash
ssh <host> "test -f <remote_repo>/<config> && echo exists || echo MISSING"
```

If missing, stop and tell the user to push locally and run `git pull` on the remote server.

## Wrapped command
- Train: `bash ./tools/dist_train.sh <config> <gpus> --deterministic`
- Test: `bash ./tools/dist_test.sh <config> <ckpt> <gpus> --deterministic --eval bbox`
- On Trillium, append `--tmpdir /tmp/.dist_test` to the test command (cwd is read-only, default `.dist_test` fails at result collection)

## Submit

### DGX
```bash
ssh trail_dgx "bash -i -c 'cd /raid/home/spapais/ForeSight && sbatch --export=ALL scripts/dgx_run.sh <wrapped_cmd>'" 2>/dev/null
```
Parse `Submitted batch job <ID>`.

### Apollo
```bash
ssh apollo "cd /home/spapais/ForeSight && ./scripts/apollo_run.sh <wrapped_cmd>"
```
Apollo runs synchronously over SSH and does not return a SLURM job ID.

### Narval
```bash
ssh narval "source ~/.bashrc && cd /home/spapais/ForeSight && sbatch --export=ALL scripts/narval_run.sh <wrapped_cmd>"
```
Parse `Submitted batch job <ID>`.

### Trillium
```bash
ssh trillium_gpu "source ~/.bashrc && cd /home/spapais/ForeSight && sbatch --export=ALL scripts/trillium_run.sh <wrapped_cmd>"
```
Parse `Submitted batch job <ID>`.

### Killarney
```bash
ssh killarney "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7 && sbatch --export=ALL /home/spapais/ForeSight/scripts/killarney_run.sh <wrapped_cmd>"
```
Parse `Submitted batch job <ID>`.

## Output
- For DGX, Narval, Trillium, and Killarney, print the job ID and suggest `/parse-metrics --server <server> --job <ID>`.
- For Apollo, warn that the session stays attached unless the user manages it with `tmux` on the remote host.

## Rules
- Always pass train or eval commands through the server wrapper script, not directly.
- DGX submission must use `bash -i` so the remote shell exports the required environment.
- Apollo has no SLURM job ID.
- Narval uses offline WandB, but metrics still land in the log files.
