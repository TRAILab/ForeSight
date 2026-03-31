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

**Run the baseline** — always submit the base config as exp-000 before any experiments. This gives a reproducible reference on the same hardware and code version.

Config stem: `auto_<tag>_exp000_baseline`

Create the config (copy base config, only update the WandB name):
```python
# === autoresearch overrides (auto_<tag>_exp000_baseline) ===
log_config['hooks'][1]['init_kwargs']['name'] = 'auto_<tag>_exp000_baseline'
```

Submit and wait for it to finish exactly as in Steps 3–6 below. Record results as the first `results.tsv` row with status `baseline`. This run does NOT count against `--max-experiments`.

**Initialize `research_log.md`** if it doesn't exist. If it already exists, read it to catch up on prior experiments before proposing.

**Read the project research review** to understand the full experimental landscape, confirmed wins, and known-bad ideas before proposing anything:
```bash
cat autoresearch/research_review.md
```
Pay particular attention to:
- **Section 3** (Experimental Findings): what has already been tried and what the results were — do NOT re-run experiments that are already documented here
- **Section 7** (Prioritized Action Plan): the "Confirmed Wins" table and "Negative/Null Evidence" table — use the confirmed wins as a starting point and never propose ideas from the null/negative list
- **Section 9** (Open Questions): unresolved questions worth answering

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
Based on the goal and all prior results in `research_log.md`, `results.tsv`, and `autoresearch/research_review.md`, decide what to change. Test ONE hypothesis per experiment (1–3 parameter changes). Explicitly state:
- What you're changing
- Why (what mechanism should improve the metric)
- What improvement you expect
- That this has NOT already been tried (cross-check `autoresearch/research_review.md` Section 3 and `research_log.md`)

**Hard constraints from prior work (never propose these — results are already in):**
- Do NOT remove map from both stages — planning catastrophically fails (L2: 0.600→6.61)
- Do NOT use pretrainv3/v4-style prediction pretraining — degrades all metrics
- Do NOT use separate head (sephead) — slightly worse across the board
- Do NOT reduce map learning rate — map_mAP collapses to ~0.07
- Do NOT add map head to stage2 when loaded from DN stage1 pretrain — L2 worsens to 0.700
- Do NOT increase motion_loss_reg or motion_loss_cls above 0.2 — large regression on both L2 and obj_box_col (mar25 exp002)
- Do NOT use rotation augmentation (rot3d_range) — hurts both L2 and obj_box_col (prior work)

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
ssh trail_dgx "cd /raid/home/spapais/ForeSight && sbatch --export=ALL,WANDB_API_KEY=1cb0a37040ca089569cecda1c31722a24d56d3a4 scripts/dgx_run.sh bash ./tools/dist_train.sh projects/configs/<config_stem>.py 4 --deterministic"
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

**Update `autoresearch/research_review.md`** — lightweight per-experiment update:
1. Append a new row to the **Section 3.11 summary table** with this experiment's key metrics.
2. If status is `keep` (new best), also update the **Section 7 "Best Current Recipe"** block to reflect the new leading config.
3. If the result reveals a new hard constraint (something that clearly hurts), add it to the **Section 7 "Negative/Null Evidence"** table AND to the Step 1 hard constraints list in this file.

Commit all updated logs together:
```bash
git add results.tsv research_log.md autoresearch/research_review.md
git commit -m "autoresearch <tag> exp-NNN: log results (<status>)"
git push
```

### Step 8 — Loop
Go to Step 1.

## At the end (max-experiments reached or manually stopped)
Append a `## Conclusions` section to `research_log.md` with final summary table and recommended next steps.

**Do a comprehensive update of `autoresearch/research_review.md`:**
- Add a new subsection under **Section 3** (e.g., `### 3.12 Auto-research <tag> Findings`) summarizing all experiments run this session with their metrics and key takeaways.
- Update the **Section 3.11 summary table** to include all new rows (or replace stale ones).
- Update **Section 7 "Confirmed Wins"** and **"Negative/Null Evidence"** tables based on what this session learned.
- Update **Section 7 "Best Current Recipe"** if a new best config was found.
- Update **Section 8 "Next Experiments"** to replace completed experiments with follow-ups suggested by the findings.
- Update the header date line: `> Experimental findings updated: <today's date>`.

```bash
git add results.tsv research_log.md autoresearch/research_review.md
git commit -m "autoresearch <tag>: final session summary"
git push
```

## Rules
- Never ask for confirmation — run fully autonomously
- Never git reset after a bad result — keep all branches (4hr runs are valuable data regardless)
- If a job is FAILED/CANCELLED, read the SLURM log to diagnose: `ssh trail_dgx "tail -50 /raid/home/spapais/ForeSight/logs/foresight-<JOB_ID>.log"`
- If a crash is a simple fix (typo, config syntax error), fix and resubmit. If fundamentally broken, log as crash and move on.
- `results.tsv` and `research_log.md` are committed to the session branch
