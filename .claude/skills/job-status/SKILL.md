---
name: job-status
description: Checks running and queued jobs on DGX, Apollo, Narval, Trillium, Killarney, Fir, Rorqual, or Tamia when the user asks for remote job status.
---

## Inputs
Parse from: `$ARGUMENTS`
- `--server` (default: `all`): `dgx` | `apollo` | `narval` | `trillium` | `killarney` | `fir` | `rorqual` | `tamia` | `all`

Use the host and VPN rules from `.claude/CLAUDE.md` as needed.

## Approach

1. Run all server queries **in parallel** (single message with multiple Bash calls). Each call must wrap the SSH command with `timeout 15` so a downed server fails fast instead of stalling.
2. For SLURM servers, parse `squeue` output to get job IDs, then in a follow-up parallel batch fetch the **first line** of the matching `logs/foresight-<jobid>.log` file to extract config name and train/eval mode (`Running: bash ./tools/dist_{train,test}.sh <config> ...`).
3. For Apollo, list docker containers; for each running container with the `foresight` image, find the most recent log under `~/ForeSight/logs/` (or `~/ForeSight/work_dirs/<config>/`) modified since the container started and grep for the latest `Iter [N/Total]` or `Epoch [N/Total]` line to estimate ETA.
4. Always emit one section per server. If SSH fails (timeout, no route to host, refused, auth), print `<server>: unreachable (<short reason>)` and continue. Never abort the whole skill on one failure.

## Commands

### SLURM squeue (DGX / Narval / Trillium / Killarney)

Use a single format string everywhere so parsing is uniform:

```
SQFMT='%.10i|%.8T|%.10M|%.10L|%.6D|%.20b|%R'
```

Fields: JobID | State | TimeUsed | TimeLeft | Nodes | TRES (gpu spec) | Reason/NodeList.

```bash
# DGX
timeout 15 ssh trail_dgx "source ~/.bashrc && squeue -u spapais --noheader --format='%.10i|%.8T|%.10M|%.10L|%.6D|%.20b|%R'" 2>&1

# Narval
timeout 15 ssh narval "source ~/.bashrc && squeue -u spapais --noheader --format='%.10i|%.8T|%.10M|%.10L|%.6D|%.20b|%R'" 2>&1

# Trillium (must use GPU login node)
timeout 15 ssh trillium_gpu "source ~/.bashrc && squeue -u spapais --noheader --format='%.10i|%.8T|%.10M|%.10L|%.6D|%.20b|%R'" 2>&1

# Killarney (needs module load)
timeout 15 ssh killarney "source /etc/profile.d/modules.sh && module load slurm/killarney/25.05.6 && squeue -u spapais --noheader --format='%.10i|%.8T|%.10M|%.10L|%.6D|%.20b|%R'" 2>&1

# Fir
timeout 15 ssh fir "source ~/.bashrc && squeue -u spapais --noheader --format='%.10i|%.8T|%.10M|%.10L|%.6D|%.20b|%R'" 2>&1

# Rorqual
timeout 15 ssh rorqual "source ~/.bashrc && squeue -u spapais --noheader --format='%.10i|%.8T|%.10M|%.10L|%.6D|%.20b|%R'" 2>&1

# Tamia
timeout 15 ssh tamia "source ~/.bashrc && squeue -u spapais --noheader --format='%.10i|%.8T|%.10M|%.10L|%.6D|%.20b|%R'" 2>&1
```

### Per-job config + mode (SLURM)

For each job ID returned above, run **in parallel**:

```bash
# DGX repo path
timeout 10 ssh trail_dgx "head -1 /raid/home/spapais/ForeSight/logs/foresight-<JOBID>.log 2>/dev/null"
# Standard Alliance Canada servers (narval, trillium, killarney, fir, rorqual): /home/spapais/ForeSight/logs/foresight-<JOBID>.log
# Tamia (SciNet path): /home/s/spapais/ForeSight/logs/foresight-<JOBID>.log
```

Parse the first line `Running: bash ./tools/dist_<train|test>.sh projects/configs/<config>.py <ngpus> ...`:
- mode = `train` if `dist_train.sh`, `eval` if `dist_test.sh`
- config = basename of the `.py` path, stripped of `.py` and `projects/configs/`
- gpus = the integer following the config path (training/eval scripts both take it as positional arg)

If the log file does not exist yet (PENDING job), set config = `-`, mode = `pending`, gpus from the squeue TRES field (`gres/gpu=N` or `gpu:N`).

### Apollo (docker)

```bash
timeout 15 ssh apollo "docker ps --format '{{.ID}}|{{.Image}}|{{.Status}}|{{.Names}}|{{.RunningFor}}'" 2>&1
```

For each container whose image starts with `foresight`:

```bash
# Find the active log (most recent foresight*.log written to since container start)
timeout 10 ssh apollo "ls -t /home/spapais/ForeSight/logs/*.log 2>/dev/null | head -1"

# Pull the launch line + last iter line for ETA
timeout 10 ssh apollo "f=\$(ls -t /home/spapais/ForeSight/logs/*.log | head -1); head -1 \$f; tac \$f | grep -m1 -E 'Iter \[[0-9]+/[0-9]+\]|Epoch \[[0-9]+\]\[[0-9]+/[0-9]+\]'"
```

Extract:
- config + mode from the launch line (same parser as SLURM).
- iter progress from the `Iter [N/Total] ... eta: H:MM:SS` line — mmcv prints `eta` directly; use that as the ETA. If only `Epoch [E][N/T]` is present and no eta token, fall back to `~unknown`.

If multiple containers run different jobs, match each container to its log via `docker inspect <id> --format '{{.State.StartedAt}}'` and pick the log whose mtime is closest to (but ≥) that timestamp.

## Output

Single combined markdown table sorted by server then by state (RUNNING before PENDING):

```
| Server   | JobID | Mode  | GPUs | State    | Elapsed | ETA       | Config                                              |
```

- `Config`: truncate to 60 chars with `…` if longer.
- `ETA`:
  - SLURM RUNNING: `TimeLeft` from squeue.
  - SLURM PENDING: the queue reason in parens, e.g. `(Priority)`, `(Resources)`.
  - Apollo: parsed `eta:` from the latest iter line, or `~unknown`.
- After the table, list any unreachable servers under a `Unreachable:` heading with the short reason (e.g. `narval: no route to host`, `apollo: connection refused`).
- If every queryable server returns nothing, print `No active jobs anywhere.`
