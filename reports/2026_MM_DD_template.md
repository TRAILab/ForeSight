# [Experiment Title]

_Use `YYYY_MM_DD_short_name.md` for the final filename. Replace bracketed placeholders and delete instructional text when writing the real report._

## TODO

_Use this section for temporary tracking items during the experiment. Remove completed items and delete the section before finalizing the report._

- `[temporary task or follow-up item]`

## Abstract

_Add this section last. Write 2-4 sentences summarizing the experiment, the main result, and the recommended takeaway._

## Intro

_Briefly state the problem, why this experiment matters, and what question you are trying to answer._

[1-2 paragraphs describing the motivation.]

Question:
- `[primary experiment issue, question, or hypothesis]`

## Method

_Describe only the changes that matter for interpreting results. Keep this section high signal._

Implementation changes:
- `[model change]`
- `[loss or target change]`
- `[data / augmentation / pipeline change]`

## Results

_Use the first table for run tracking. Add metric tables below it once runs finish._

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `[server]` | `[short config name]` | `[job id]` | `[RUNNING / COMPLETED / FAILED / CANCELLED]` |

_Add one or more tables here for the metrics that matter for the experiment. Keep the baseline and best variant easy to compare._

| Model | Metric 1 | Metric 2 | Metric 3 | Notes |
| --- | --- | --- | --- | --- |
| `[baseline]` | `[x]` | `[y]` | `[z]` | `[short note]` |
| `[variant]` | `[x]` | `[y]` | `[z]` | `[short note]` |

## Discussion

_Summarize what changed and why it likely changed. Focus on actual evidence, not speculation alone._

[Short interpretation of the results.]

What improved:
- `[metric / behavior]`

What regressed or stayed flat:
- `[metric / behavior]`

Likely explanation:
- `[short explanation tied to the method]`

_If there is a clear winner, name it here and explain why it should be the default or next reference point._

Recommended model:
- `[config name]`

Why:
- `[best tradeoff or strongest result]`

## Future Work

_List the next experiments that follow naturally from the results in this report._

- `[next ablation or cleanup]`
- `[follow-up experiments]`
- `[dataset / metric / evaluation extension]`
