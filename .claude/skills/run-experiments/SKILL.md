---
name: run-experiments
description: Plans, creates, submits, monitors, and logs a small ForeSight experiment batch after surveying prior results and waiting at two confirmation checkpoints.
allowed-tools: Read, Write, Edit, Bash(git:*), Bash(ssh:*), Bash(cp:*), Bash(date:*), Bash(cat:*), Bash(head:*), Bash(tail:*), Bash(ls:*)
---

Goal: `$ARGUMENTS`

Default assumptions unless the goal says otherwise:
- server: `narval`
- base config: `projects/configs/sparsedrive_r50_stage2_4gpu_bs24.py`
- experiment count: 1 to 3 independent experiments

All experiments must target the stated goal and the primary planning metrics first.

## Phase 1: Survey
Read:
```bash
cat reports/2026_02_16_baselines.md
cat reports/0000_00_00_research_findings.md
ls -1 reports/
```

If the goal references an existing experiment batch report (e.g. `plan_relevance`, `plan_aux`, `plan_unified`), read that report too — it likely contains the method, planned experiments, and decisions already agreed with the user.

Then read only the relevant architecture files and report files for the goal.

## Phase 2: Propose
Write:

```text
Problem: <one sentence>
Approach: <one sentence>
```

Then design 1 to 3 experiments in this format:

| # | Config suffix | Change | Mechanism | Expected effect |
|---|---------------|--------|-----------|-----------------|
| 1 | `<suffix>` | `<old> -> <new>` | `<why>` | `<metric direction>` |

Rules:
- Each experiment tests one hypothesis.
- Experiments are independent.
- Do not repeat ideas already ruled out in `reports/0000_00_00_research_findings.md`.

### Checkpoint 1
Print the problem statement, approach, and full experiment table. Ask exactly:
`Does this look correct? Reply 'yes' to proceed or give feedback to revise.`

Do not continue until the user confirms.

## Phase 3: Create
1. Derive `<feature>` as a short snake_case label and work on branch `sd_<feature>`, unless the goal specifies a branch name.
2. Report file: check whether `reports/<YYYY_MM_DD>_<feature>.md` already exists.
   - If it exists: read it, then update only the `Method` section (and `TODO` if present) to reflect the confirmed experiment plan. Do not replace content that is already correct.
   - If it does not exist: create it with title `<feature> - <YYYY-MM-DD>` and sections `Intro`, `Method`, `Results`, `Discussion`, `Future Work`, with `Intro` and `Method` filled in now.
3. If the experiments require source code changes (e.g. new loss heads, new cross-attention blocks, new dataset keys):
   - Implement the changes in the relevant source files (`detection3d_head.py`, `motion_planning_head.py`, `nuscenes_3d_dataset.py`, etc.) before creating configs.
   - Keep each change minimal and scoped to the experiment — no refactoring.
4. For each experiment:
   - copy the base config to `projects/configs/<config_stem>.py`
   - edit only the lines that must differ, in place
   - set the WandB run name to `<config_stem>`
5. Re-read every new config and confirm the intended changes only.

### Checkpoint 2
Summarize all source code changes and config changes. Ask exactly:
`Ready to commit and push? Reply 'yes' to proceed or give feedback to revise.`

Do not commit or push until the user confirms.

6. Commit, push, and sync the target server:
```bash
git add projects/configs/<config_stem_1>.py projects/configs/<config_stem_2>.py ... <changed source files>
git commit -m "sd_<feature>: add experiment configs

<brief description>"
git push -u origin sd_<feature>
ssh <server> "source ~/.bashrc && cd /home/spapais/ForeSight && git fetch origin sd_<feature> && git checkout sd_<feature>"
```

## Phase 4: Submit
Submit every config with `/submit-job --server <server> --config projects/configs/<config_stem>.py`.

Record each returned job ID in `Results`.

## Phase 5: Triage
State handling:
- `COMPLETED`: mark done
- `FAILED`: inspect the log, fix the root cause, resubmit once
- `TIMEOUT`: treat like `FAILED`
- `CANCELLED`: mark as crash and do not resubmit

For failed jobs, inspect:
```bash
ssh <server> "tail -80 /home/spapais/ForeSight/logs/foresight-<JOB_ID>.log"
```

If a non-trivial fix is required, ask for confirmation before commit/push, then resubmit once. Never resubmit the same experiment more than once.

## Phase 6: Log Results
1. Run `/parse-metrics --server <server> --job <JOB_ID> --config <config_stem>` for every completed experiment.
2. Update `Results`, `Discussion`, and `Future Work`.
3. Update `reports/0000_00_00_research_findings.md`.
4. Ask exactly:
`Ready to commit and push the logged results? Reply 'yes' to proceed or give feedback to revise.`
5. Do not commit or push the logged results until the user confirms.
6. Commit the logging updates:
```bash
git add <report> reports/0000_00_00_research_findings.md
git commit -m "sd_<feature>: log results"
git push
```
7. Print a concise experiment summary table and one-sentence conclusion.

## Rules
- Use `/submit-job` for remote train or eval submission.
- Use `/parse-metrics` for final metric extraction.
- Ask for confirmation before every commit/push step.
- Never discard branches because results are poor.
- Keep all session logging in `reports/<YYYY_MM_DD>_<feature>.md`.
