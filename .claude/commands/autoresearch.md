You are running an autonomous ML research loop for the ForeSight autonomous driving project.

## Setup

Arguments: $ARGUMENTS
Parse:
- `--goal` (required): research objective
- `--base-config` (default: `projects/configs/sparsedrive_r50_stage2_4gpu.py`)
- `--max-experiments` (default: 5)
- `--poll` (default: 30m): how often to check job status (e.g. `30m`, `1h`)

**Agree on a run tag** based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist.

```bash
git checkout -b autoresearch/<tag>
```

Read the base config for full context before proposing anything:
```bash
head -90 <base-config> && echo "---" && tail -50 <base-config>
```

**Initialize `results.tsv`** if it doesn't exist (tab-separated, NOT comma-separated):
```
commit	val_L2	val_col%	car_ade	NDS	status	description
```

**Establish the baseline**: check whether prior experiment logs exist in `work_dirs/` on DGX for the base config. If they do, read the metrics and record them as the first `results.tsv` row with status `baseline`. If not, note that no baseline is available and proceed.

**Initialize `research_log.md`** if it doesn't exist. If it already exists, read it to catch up on prior experiments before proposing.

## Architecture
- SparseDrive: ResNet → FPN → SparseDriveHead (detection + map + motion/planning)
- Configs are Python files exec()'d by mmdet3d — appending lines at the end overrides earlier values
- Training: 4 GPUs on DGX server (ssh host: `trail_dgx`), repo at `/raid/home/spapais/ForeSight`
- Each experiment takes ~4 hours

## Key Metrics (nuScenes val)
- **L2**: ego planning L2 error in meters (lower = better) ← primary metric
- **obj_box_col**: planning collision rate % (lower = better) ← primary metric
- **car_ade / ped_ade**: agent motion ADE in meters (lower = better)
- **car_epa / ped_epa**: motion end-point accuracy (higher = better)
- **NDS**: nuScenes detection score (higher = better)
- **mAP**: detection mean AP (higher = better)
- **mAP_normal**: map prediction mAP (higher = better)

## Tunable Parameters

**Top-level variables** (simple reassignment appended to config):
```python
num_decoder = 6                  # transformer decoder layers (2–8)
num_single_frame_decoder = 1
embed_dims = 256                 # feature embedding dim (128/256)
num_groups = 8                   # attention heads
drop_out = 0.1                   # dropout (0–0.3)
num_epochs = 10
queue_length = 4                 # temporal history frames (1–6)
fut_ts = 12                      # motion future timesteps
ego_fut_ts = 6                   # planning future timesteps
temporal = True
decouple_attn_motion = True
```

**Nested params** (dict mutation appended to config):
```python
optimizer['lr'] = 3e-4
model['depth_branch']['loss_weight'] = 0.2
model['head']['motion_plan_head']['motion_loss_cls']['loss_weight'] = 0.2
model['head']['motion_plan_head']['motion_loss_reg']['loss_weight'] = 0.2
model['head']['motion_plan_head']['plan_loss_cls']['loss_weight'] = 0.5
model['head']['motion_plan_head']['plan_loss_reg']['loss_weight'] = 1.0
model['head']['motion_plan_head']['plan_loss_status']['loss_weight'] = 1.0
model['head']['det_head']['loss_cls']['loss_weight'] = 2.0
model['head']['det_head']['loss_reg']['loss_box']['loss_weight'] = 0.25
```

## Experiment Loop

**NEVER STOP.** Once the loop begins, do NOT pause to ask whether to continue. The user may be away or asleep and expects you to run until manually stopped or max-experiments is reached. If you run out of obvious ideas, think harder — re-read prior results, try combining near-misses, try more radical changes. Keep going.

Each iteration:

### Step 1 — Propose
Based on the goal and all prior results in `research_log.md` and `results.tsv`, decide what to change. Test ONE hypothesis per experiment (1–3 parameter changes). Explicitly state:
- What you're changing
- Why (what mechanism should improve the metric)
- What improvement you expect

### Step 2 — Create config
Config stem format: `auto_<tag>_exp{NNN}_{short_suffix}` (suffix: alphanumeric+underscore, ≤20 chars)

```bash
git checkout autoresearch/<tag>  # ensure we're on the session branch
```

Use the **Write tool** to create `projects/configs/<config_stem>.py`:
- Copy the full base config content
- Append at the end:
```python

# === autoresearch overrides (<config_stem>) ===
log_config['hooks'][1]['init_kwargs']['name'] = '<config_stem>'
<your parameter changes>
```

### Step 3 — Commit, push, sync DGX
```bash
git add projects/configs/<config_stem>.py
git commit -m "autoresearch <tag> exp-NNN: <suffix>

<reason>"
git push -u origin autoresearch/<tag>
ssh trail_dgx "cd /raid/home/spapais/ForeSight && git fetch origin autoresearch/<tag> && git checkout autoresearch/<tag>"
```

### Step 4 — Submit
```bash
ssh trail_dgx "cd /raid/home/spapais/ForeSight && sbatch scripts/dgx_run.sh bash ./tools/dist_train.sh projects/configs/<config_stem>.py 4 --deterministic"
```
Parse job ID from `Submitted batch job <ID>`. Record it immediately in `research_log.md`.

### Step 5 — Wait
Use the `/loop` skill to poll for job completion at the `--poll` interval:
```
/loop <poll> Check if SLURM job <JOB_ID> is done: ssh trail_dgx "squeue -j <JOB_ID> -h -o %T 2>/dev/null" — if the output is empty the job has finished; when done, continue the autoresearch loop by parsing metrics for config <config_stem> (tag <tag>, exp-NNN)
```
The loop will wake Claude every `--poll` interval. When the job leaves the queue, Claude continues automatically to Step 6.

### Step 6 — Parse metrics
```bash
LOG=$(ssh trail_dgx "ls -t /raid/home/spapais/ForeSight/work_dirs/<config_stem>/*.log 2>/dev/null | head -1")
ssh trail_dgx "cat $LOG" | grep -E "NDS|mAP|ade=|epa=|L2|obj_box_col|mAP_normal" | tail -30
```
If no work_dir log, fall back to SLURM log: `/raid/home/spapais/ForeSight/logs/foresight-<JOB_ID>.log`

### Step 7 — Log results

**Determine status:**
- `keep` — primary metrics (L2, obj_box_col) improved vs best so far
- `discard` — no improvement (but keep the branch — 4hr runs are worth recording)
- `crash` — job failed or no metrics found

**Append to `results.tsv`** (tab-separated):
```
<short-commit>	<L2>	<col%>	<car_ade>	<NDS>	<status>	<short description>
```

**Append to `research_log.md`:**
```markdown
## [exp-NNN] <config_stem> — <date>
**Hypothesis:** <what and why>
**Config changes:**
\`\`\`python
<appended lines>
\`\`\`
**Job ID:** <id>
**Status:** keep / discard / crash
**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | x.xx | x.xx | x.xx | ↓/↑ |
| obj_box_col | x.xx | x.xx | x.xx | ↓/↑ |
| car_ade | x.xx | x.xx | x.xx | ↓/↑ |
| NDS | x.xx | x.xx | x.xx | ↓/↑ |
**Analysis:** <what this tells us, what to try next>
---
```

Commit the updated logs:
```bash
git add results.tsv research_log.md
git commit -m "autoresearch <tag> exp-NNN: log results (<status>)"
git push
```

### Step 8 — Loop
Go to Step 1.

## At the end (max-experiments reached or manually stopped)
Append a `## Conclusions` section to `research_log.md` with final summary table and recommended next steps. Commit and push.

## Rules
- Never ask for confirmation — run fully autonomously
- Never git reset after a bad result — keep all branches (4hr runs are valuable data regardless)
- If a job is FAILED/CANCELLED, read the SLURM log to diagnose: `ssh trail_dgx "tail -50 /raid/home/spapais/ForeSight/logs/foresight-<JOB_ID>.log"`
- If a crash is a simple fix (typo, config syntax error), fix and resubmit. If fundamentally broken, log as crash and move on.
- `results.tsv` and `research_log.md` are committed to the session branch
