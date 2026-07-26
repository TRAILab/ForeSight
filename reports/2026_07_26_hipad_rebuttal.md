# HiP-AD Interface Ablation — Rebuttal Plan

Date: 2026-07-26
Repo: `HiP-AD` (ICCV'25, nullmax-vision). Baseline reproduced from released `hipad_stage2.pth`.

---

## 1. Goal

Show that cutting the perception→planner interface does not hurt driving performance, on an
architecture we did not design, in closed loop, on a second dataset.

HiP-AD concatenates det/map/plan/ego queries into one sequence and runs joint self-attention
(`gnn`, `inter_gnn`) over it — `inter_gnn` (distance/velocity-gated planning↔perception
interaction) is their headline contribution. It is also a **Parallel** architecture
(DriveTransformer-style), not Sequential like SparseDrive, so it extends our evidence to a class
we have not tested.

**Token layout:** `query_select = ["det", "map", "plan", "ego"]` → **900 / 100 / 48 / 1 = 1049**.
Plan count derives from `ego_fut_mode=48` with `with_concat_plan_points=False`.

**Plan → image cross-attention is untouched by the mask.** The mask reaches only `graph_model`,
called from `temp_gnn` (`:815`), `gnn` (`:831`), `inter_gnn` (`:844`). Image cross-attention is a
separate op that takes no `attn_mask` and runs *after* `split`, on per-modality features:

```
concat → temp_gnn → gnn → inter_gnn → norm → split → deformable → concat → ffn → norm → split → refine
                    └──── masked ────┘                └─ images ─┘
```

So the masked arm is exactly the architecture our paper describes: planner reads image features
directly via deformable sampling at trajectory waypoints, with zero perception tokens entering it.

---

## 2. Code changes

### Config (`projects/configs/hipad_b2d_stage2.py`)

Add inside the model dict, alongside `task_select` / `query_select` (~line 147):

```python
with_attn_mask = True,
attn_mask_dict = dict(
    det  = ["det", "map", "plan", "ego"],   # untouched
    map  = ["det", "map", "plan", "ego"],   # untouched
    plan = ["plan", "ego"],                 # perception interface cut
    ego  = ["plan", "ego"],                 # perception interface cut
),
```

**The dict must be written out in full.** Construction at `sparse_onedecoder.py:588` fills `-inf`
then zeros only the listed pairs — any omitted pair is cut, and an empty dict (the default
`dict()`) masks everything and yields NaN. This also cuts the temporal perception→plan edges via
`temp_attn_mask` (`:600`), which is intended.

### Guard (`projects/mmdet3d_plugin/models/sparse_onedecoder.py`, ~`:583`)

Both masks are built once under `if self.attn_mask is None` and cached. Two risks, invisible
today because `with_attn_mask=False` ships as default:

- **Shape staleness.** `num_temp_ego_anchor` / `num_temp_plan_anchor` are set per-forward from
  actual tensor sizes (`:493`, `:534`), so `total_num_temp_anchor` varies between the first frame
  of a sequence and later ones.
- **All-`-inf` rows → NaN.** In the masked arm, the plan/ego rows of the *temporal* mask attend
  only to temp plan+ego columns. If both are zero-width, softmax returns NaN. The main mask is
  safe (plan+ego always ≥49 wide); the temporal one is not.

```python
# force rebuild if the cached temporal mask no longer matches
if self.temp_attn_mask is not None and \
   self.temp_attn_mask.shape != (self.total_num_anchor, self.total_num_temp_anchor):
    self.temp_attn_mask = None

# after building, assert no row is fully masked
assert (self.temp_attn_mask > float("-inf")).any(dim=1).all()
```

Not a concern: the top-k prune at `:1006` that rewrites `num_anchor_list` mid-forward is gated on
`with_topk_mode`, which defaults `False` (`:150`) and is not set in the stage-2 config. Dead path
here; the anchor layout is stable.

---

## 3. Experiments

Two arms, fine-tuned from the released checkpoint under an identical schedule, differing only in
the mask.

| Run | Arm | Recipe |
|---|---|---|
| **R1** | unmasked (control) | 2 epochs, lr 2e-5, frozen backbone+neck |
| **R2** | masked (`plan`/`ego` ← `plan`, `ego`) | identical |

Report **R2 − R1**. Comparing R2 against the released checkpoint is confounded by the extra
epochs at restarted LR, so R1 is not optional.

**Optional third arm** if R2 degrades and we want to localize which channel matters: mask only det
or only map (`plan`/`ego` ← `["map","plan","ego"]` or `["det","plan","ego"]`). Our pattern is that
map is dispensable and det is sticky (`laststage_nomapkv` tied, `nodetkv` regressed), so a
map-only cut is the likely survivor. Another ~5.2h.

### Recipe

Their stage 2: 8×6 = global batch 48, 4891 iters/epoch, 18 epochs = 88,038 iters, 46h on 8×4090 →
**~1914 iters/h**. Check observed it/s in the first ~100 iters and rescale.

| Param | Stage 2 | Fine-tune | Why |
|---|---|---|---|
| GPUs × batch | 8 × 6 | **4 × 12** | preserves global batch 48 on 80GB |
| Epochs | 18 | **2** | ~11% of original; recalibration, not representation learning |
| Peak lr | 2e-4 | **2e-5** | checkpoint sits at end of cosine decay (2e-7) |
| Warmup | 500 iters | **200** | proportional |
| `img_backbone` | `lr_mult=0.5` | **`lr_mult=0.0`** | frozen; also freeze `img_neck` |
| grad clip / fp16 / seq | — | unchanged | no free variables |

**Cost: ~5.2h per experiment.**

```python
num_gpus = 4
batch_size = 12
num_epochs = 2
load_from = "ckpts/hipad_stage2.pth"   # NOT resume_from
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)

optimizer = dict(
    type="AdamW", lr=2e-5, weight_decay=0.001,
    paramwise_cfg=dict(custom_keys={
        "img_backbone": dict(lr_mult=0.0),
        "img_neck":     dict(lr_mult=0.0),
    }),
)
lr_config = dict(policy="CosineAnnealing", warmup="linear",
                 warmup_iters=200, warmup_ratio=1.0/3, min_lr_ratio=1e-3)
```

- **`load_from`, not `resume_from`.** `resume_from` restores the iteration counter; since
  `max_iters` is already exceeded the run exits having trained nothing.
- `max_keep_ckpts=3` gives the 1-epoch checkpoint free — two duration points for one run cost.
- Verify frozen params show zero grad norm in the log. `paramwise_cfg` matches by substring and
  depends on module attribute names in `sparse_detector.py`.
- Watch training loss over the first ~200 iters. A spike above the checkpoint's converged level
  means lr is too hot — kill and fall back to lr 5e-6 on both arms.
- One recipe, applied identically to both arms. Whatever it does, it does to both and largely
  cancels in the delta.

### Leave alone

- **Perception losses stay on.** `rescore()` reads `det_output["prediction"][-1]` and
  `["classification"][-1]` directly (`plan/decoder.py:301-303`) with `with_rescore=True` in the
  config, so collapsing det feeds the rescore garbage boxes and breaks mode selection. It is also
  a no-op under this recipe: with the backbone frozen and plan/ego not attending to det/map, the
  perception heads receive ~zero gradient and do not move.
- **`with_target_point_embed` stays ON** (`:154`). `target_point` is the route-following
  navigation goal and bypasses attention entirely (`plan_anchor_embed += ...`, `:955`). Removing
  it collapses driving score for reasons unrelated to the claim.

---

## 4. Evaluation

- **Closed loop:** Bench2Drive **Dev10**
  (`bench2drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml`) — 10 routes selected by
  the DriveTransformer authors for low-variance ablation. Much cheaper than the 220-route set.
- **Open loop:** `datasets/evaluation/planning/metric_stp3.py` on b2d val — cheap direction check
  on both arms before spending CARLA compute.
- **Run both arms twice on Dev10.** CARLA is stochastic and Dev10 is 10 routes; we have no eval
  noise floor yet, and a delta without a spread is not interpretable.
- Each arm also has a 1-epoch checkpoint from `max_keep_ckpts=3` — free second data point on
  whether the result is sensitive to adaptation length.

---

## 5. Checklist

| Work | Gate |
|---|---|
| Guard + config changes | — |
| Smoke test: forward R2's config on a few training batches | `num_anchor_cumsum == [0,900,1000,1048,1049]`; `attn_mask.shape == (1049,1049)`; masked block `-inf`, kept block `0`; no NaN in first ~20 iters; frozen params at zero grad norm. **Stop if any fail** |
| Launch R1 + R2 (~10.4h; parallel if 8 GPUs, else serial) | Loss spike in first ~200 iters → fall back to lr 5e-6 on both arms |
| Open-loop eval, both arms | Direction check before spending CARLA compute |
| Dev10 closed loop, both arms, ×2 | — |
| Collect DS / SR / infraction breakdown | — |
