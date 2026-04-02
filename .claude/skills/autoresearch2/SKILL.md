---
name: autoresearch2
description: One-shot ML experiment batch for ForeSight — survey prior work, propose a batch of experiments, create all configs, queue all jobs on SLURM, then log all results. Two mandatory confirmation checkpoints before any compute is used.
allowed-tools: Read, Write, Edit, Bash(git:*), Bash(ssh:*), Bash(date:*), Bash(cat:*), Bash(head:*), Bash(tail:*), Bash(ls:*)
---

Arguments: $ARGUMENTS
Parse:
- `--goal` (required): what is underperforming or missing functionality
- `--base-config` (default: `projects/configs/sparsedrive_r50_stage2_4gpu.py`)
- `--server` (default: `narval`)

Objective: improve key metrics below. All experiments must serve this goal.
- **L2**: ego planning L2 error in meters (lower = better) ← primary metric
- **obj_box_col**: planning collision rate % (lower = better) ← primary metric
- **car_ade / ped_ade**: agent motion ADE in meters (lower = better)
- **car_epa / ped_epa**: motion end-point accuracy (higher = better)
- **NDS**: nuScenes detection score (higher = better)
- **mAP**: detection mean AP (higher = better)
- **mAP_normal**: map prediction mAP (higher = better)

---

## Phase 1 — Survey

### 1. Read the base config

Read the full base config to understand the current model setup:

```bash
cat <base-config>
```

### 2. Review the model architecture

If `--goal` involves a specific component (detection, map, motion, or planning), also read the relevant source file from the **Key Source Files** table:
| File | Purpose |
|------|---------|
| `projects/mmdet3d_plugin/models/sparsedrive_head.py` | Main head dispatcher |
| `projects/mmdet3d_plugin/models/detection3d/detection3d_head.py` | Sparse4DHead |
| `projects/mmdet3d_plugin/models/motion/motion_planning_head.py` | MotionPlanningHead |
| `projects/mmdet3d_plugin/models/motion/instance_queue.py` | Temporal tracking queue |
| `projects/mmdet3d_plugin/models/instance_bank.py` | Instance feature bank |
| `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py` | Dataset + evaluation |

### 3. Read prior research findings

**Always** read the baseline performance report:
```bash
cat reports/2026_02_16_baselines.md
```

From the baseline report, extract the specific metric values for each experiment (use these as your baseline reference).

Then list all available reports and read the ones relevant to `--goal`:

```bash
ls -1 reports/
```
From each report, extract: which changes worked and which did not, and any failure modes documented.

---

## Phase 2 — Propose

### 1. State the problem

Based on `--goal` and all survey findings:

> **Problem:** `<one sentence — what is underperforming or missing functionality>`
> **Approach:** `<one sentence — the improvements these experiments will test>`

### 2. Propose all experiments

Design 1-3 experiments. Each must test **one independent hypothesis** (1-3 parameter changes). Present them as a table:

| # | Config suffix | Change | Mechanism | Expected effect |
|---|--------------|--------|-----------|-----------------|
| 1 | `<suffix_1>` | `<param>: <old> → <new>` | `<why this helps>` | `<metric direction>` |
| 2 | `<suffix_2>` | ... | ... | ... |
| N | `<suffix_N>` | ... | ... | ... |

Rules:
- Experiments are independent — no experiment depends on the result of another in this batch
- Propose experiments that are most likely to improve the primary metrics

---

**CHECKPOINT 1:** Print the problem statement and the full experiment table. Ask: "Does this look correct? Reply 'yes' to proceed or give feedback to revise." Do not proceed until confirmed.

---

## Phase 3 — Create

### 1. Create the session branch

Derive `<feature>` from the goal: a short snake_case label (e.g. `planning_loss`, `temporal_queue`, `map_stability`).

```bash
git checkout -b sd_<feature>
```

### 2. Create the session report

```bash
date +%Y_%m_%d
```

Create `reports/<YYYY_MM_DD>_<feature>.md` (referred to as `<report>` throughout). This file accumulates all logging for the session:

```markdown
# <feature> — <YYYY-MM-DD>

## Intro
Describe the problem and the experiments.

## Method

Describe the method used to test the hypotheses.

## Results

Create a table with the results of the baseline and experiments:

| # | Config | L2 | obj_box_col | car_ade | NDS | Status | Notes | Job ID |
|---|--------|----|-------------|---------|-----|--------|-------|--------|

## Future Work
_(filled at the end)_
```

### 3. Create all configs

`<config_stem>` = `<base_config_stem>_<suffix>` where `<base_config_stem>` is the filename stem of `<base-config>` (e.g. `sparsedrive_r50_stage2_4gpu`) and `<suffix>` matches the suffix column from the proposal table.

For **each** experiment:
1. Use the **Shell tool** to copy the base config: `cp <base-config> projects/configs/<config_stem>.py`
2. Use the **StrReplace tool** to make each targeted change in place — edit only the lines that differ from the base config
3. Also update the WandB run name in place: find the `name=` line inside `log_config` and replace it with `'<config_stem>'`

After editing all configs, re-read each one to confirm every change is correct and no unintended lines were modified.

---

**CHECKPOINT 2:** List all config files to be committed. Ask: "Ready to commit and push? Reply 'yes' to proceed or give feedback to revise." Do not commit or push until confirmed.

---

### 4. Commit, push, and pull on server

```bash
git add projects/configs/<config_stem_1>.py projects/configs/<config_stem_2>.py ...
git commit -m "sd_<feature>: add experiment configs

<brief description of what is being tested>"
git push -u origin sd_<feature>
ssh <server> "source ~/.bashrc && cd /home/spapais/ForeSight && git fetch origin sd_<feature> && git checkout sd_<feature>"
```

---

## Phase 4 — Submit

Submit all jobs using the `/submit-job` skill. Run one submission per config:

```
/submit-job --server <server> --config projects/configs/<config_stem_1>.py
/submit-job --server <server> --config projects/configs/<config_stem_2>.py
...
```

Record all returned job IDs in the **Job IDs** table in `<report>`.

---

## Phase 5 — Poll & Triage

Poll until **all** jobs have either completed or been written off. Use the `/loop` skill at a 1h interval:

```
/loop 1h Check SLURM job statuses: ssh <server> "squeue -j <JOB_ID_1>,<JOB_ID_2>,... -h -o '%i %T' 2>/dev/null" and ssh <server> "sacct -j <JOB_ID_1>,<JOB_ID_2>,... --format=JobID,State --noheader 2>/dev/null" — triage any newly finished or failed jobs per the Phase 5 rules, then if all jobs are resolved continue the autoresearch2 skill by logging results for branch sd_<feature>
```

On each wake, for every job that has left `squeue`, check its final state via `sacct` and apply the following:

| `sacct` State | Action |
|---------------|--------|
| `COMPLETED` | Mark done. |
| `FAILED` | Diagnose and fix (see below). |

**For every `FAILED` job**, tail the log to find the cause:

```bash
ssh <server> "tail -80 /home/spapais/ForeSight/logs/foresight-<JOB_ID>.log"
```

Fix the root cause — whether it's a config typo, import error, assertion, or code bug — then resubmit:

```
/submit-job --server <server> --config projects/configs/<config_stem>.py
```

Update the Job IDs table in `<report>` with the new job ID. If the fix requires a non-trivial code change, commit it before resubmitting:

```bash
git add <changed files>
git commit -m "sd_<feature>: fix <config_stem> (<one-line error>)"
git push && ssh <server> "source ~/.bashrc && cd /home/spapais/ForeSight && git pull"
```

Each job may be resubmitted **at most once**. If a resubmitted job also fails, mark it as `crash` and move on.

When all jobs are either `COMPLETED` or written off as `crash`, proceed to Phase 6.

---

## Phase 6 — Log Results

Process **all** experiments. Do not skip any, even if they crashed.

### 1. Parse metrics for each experiment

```
/parse-metrics --server <server> --job <JOB_ID> --config <config_stem>
```

If a job was marked `crash` during Phase 5 triage, skip metric parsing and use the error summary already recorded in `<report>`.

### 2. Update `<report>` for each experiment

Add a subsection per experiment and fill its row in the Experiments table:

```markdown
### <config_stem>
**Hypothesis:** <what and why>
**Config changes:**
\`\`\`python
<appended lines>
\`\`\`
**Status:** keep / discard / crash
**Metrics:**
| Metric | Baseline | This exp | Δ |
|--------|----------|----------|---|
| L2 | x.xx | x.xx | ↓/↑ |
| obj_box_col | x.xx | x.xx | ↓/↑ |
| car_ade | x.xx | x.xx | ↓/↑ |
| NDS | x.xx | x.xx | ↓/↑ |
**Analysis:** <what this tells us>
```

Status key:
- `keep` — primary metrics (L2, obj_box_col) improved vs baseline
- `discard` — no improvement
- `crash` — job failed or no metrics found

### 3. Fill `## Conclusions` in `<report>`

Write a cross-experiment summary: which hypotheses were confirmed, which were not, and recommended next steps.

### 4. Update `reports/0000_00_00_research_review.md`

- Add new entries to **Confirmed Findings** for results that extend what is known.
- Update **Promising Directions** "Highest priority" block if a new best was found.
- Add any newly confirmed failure modes to **Avoid / treat carefully**.
- Update the `Last updated:` date line.

### 5. Commit all logs

```bash
git add <report> reports/0000_00_00_research_review.md
git commit -m "sd_<feature>: log results"
git push
```

### 6. Print a summary

Print a concise table of all experiments with their key metrics and status. State the overall conclusion in one sentence.

---

## Rules
- Never ask for confirmation outside of the two designated checkpoints
- Never git reset after a bad result — keep all branches regardless of outcome
- All session logging goes into `<report>` (`reports/<YYYY_MM_DD>_<feature>.md`)
- Each experiment must test a single independent hypothesis
- Never propose ideas already documented in `reports/0000_00_00_research_review.md` Confirmed Findings or Avoid lists
