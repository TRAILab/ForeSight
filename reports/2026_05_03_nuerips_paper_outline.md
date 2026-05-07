# NeurIPS '26 Paper Outline

**Working title:** *Rethinking Sparse Scene Representations for End-to-End Driving*

Date: 2026-05-03

---

## One-sentence pitch

End-to-end driving stacks (UniAD, VAD, SparseDrive) feed detection / map / motion tokens into the planner at inference — we show that interface is essentially **empty**, and perception heads only matter as **training-time supervision**. The planner reads what it needs straight from image features.

## The reframe

Sparse perception plays *two* conflated roles:

1. **Supervision** — gradients that shape the backbone.
2. **Interface** — tokens piped into the planner at inference.

The field treats both as essential. We separate them empirically and show that **the supervision is load-bearing; the interface is not**. Stage 1 needs perception decoders (as auxiliary supervisors). Stage 2 does not need them on the inference path at all.

## Contributions

**(i) Perception in end-to-end driving is supervision, not interface.** The perception-to-planner channel at inference carries no measurable information, while removing perception supervision at training collapses planning by an order of magnitude.

**(ii) We propose an architecture that operationalizes this decoupling.** Perception decoders supervise the visual backbone during training but are removed from the inference path entirely; the planner reads directly from image features, with no perception, map, or agent tokens entering the planning decoder.

**(iii) State of the art on nuScenes camera-based end-to-end planning** — L2 = 0.37 m, CR = 0.04% (3-seed mean) — with a strictly simpler inference graph than every prior method.

## Positioning

Two recent trends in end-to-end driving offer complementary benefits at complementary costs.

- **SSL / world-model methods (SSR, LAW)** remove perception from the inference path, but require new self-supervised training objectives and learned token bottlenecks — not retrofittable to existing perception-trained pipelines.
- **Multi-task parallel architectures (DriveTransformer, ParaDrive)** preserve standard perception supervisors and exploit shared backbone representations, but keep all task heads coupled through shared queries or joint self-attention at inference.

Our method combines the inference simplicity of the first with the shared-supervision benefits of the second, with **neither a world-model substitute nor inference-time task coupling**. The two axes — *what supervises the backbone* and *what enters the planner at inference* — have historically been tied together; we untie them. The result is a drop-in retrofit to existing perception-trained stacks that uses no world-model auxiliaries, no learned compression, and no inference-time coupling between tasks.

We make two architectural simplifications, supported at different evidence levels:

1. **Intermediate detection, map, and motion tokens add little to planning at inference** once camera features are directly accessible — yet their training losses remain critical for shaping planning-relevant visual features. Supported by controlled ablation (token-count grid, loss-zeroing trio, frozen-backbone diagnostic).
2. **The BEV lift itself is unnecessary** — multi-view camera features sampled along the predicted trajectory provide sufficient spatial grounding without an intermediate BEV transformation. Supported by architectural design and concurrent evidence (LAW, DriveTransformer reach SOTA without BEV).

The first claim is the load-bearing contribution; the second is consistent with an emerging consensus in concurrent work.

## Headline numbers (locked across 3 seeds)

- **K/V-off paper headline:** L2 = 0.369, CR = 0.040%
- **minS2** (everything off in S2 except ego query + planning loss): L2 = 0.356, CR = 0.041% — ties or marginally beats headline (single seed).
- Beats every prior K/V-on result in this codebase on both L2 and CR.

## Why it's interesting

- ~15 controlled negatives across token count (0/25/50/100/900), selection criterion (top-k, oracle, learned), and bidirectional flow — all null or worse. The interface is *saturated*.
- Stage-2 perception losses (det / map / motion, individually and combined) all zero out without hurting planning.
- A fully frozen S2 backbone + perception heads ties the headline.
- Dropping all agent slots from the planner (ego-only) ties or beats.
- Map-K/V dispensable. Det-K/V dispensable when paired with a learned rescore.

→ Stage 2 is just *"fine-tune the planner on a well-shaped backbone."*

## Key Insights

1. **Hidden redundancy in sparse-scene interfaces.** A controlled ablation grid (token count, selection, bidirectional flow, supervision/coupling) shows the perception → planner K/V channel has no measurable effect on planning.
2. **The real bottleneck is image-feature alignment.** Every confirmed positive lever in the planning era touches the image-feature path; every perception-token-path lever has been null or regressive.
3. **Decoupling supervisory and interface roles.** Stage-1 nomap + stage-2 with map preserves planning; both-stage nomap is catastrophic. The load-bearing role of perception is stage-1 backbone shaping, not stage-2 inference.
4. **A reusable abstraction: scene representations as supervision without interface.** Concrete instantiation = K/V-off stage-2 architecture. Perception heads kept for training-time gradient flow only (or eval reporting).
5. **Simplify without sacrificing performance.** Direct visual planning matches or improves over sparse-scene-interface baselines while retaining rich training supervision. Lower latency, fewer moving parts, same or better L2/CR.

---

## Architecture

### Stage 1 — backbone shaping

Stage 1 is a multi-task representation-learning problem. Perception decoders earn their keep here — their gradients shape the image features the planner will later read.

- **Det + map + motion heads** (kept) — gradient pressure on the backbone. Det K/V is *not* fed forward into stage 2's planner.
- **`aux2d` depth supervision** (added) — dense per-pixel depth aux on the FPN features. Single biggest non-perception lever for planning L2.
- **(Pending) `dseg` v2** — dense BEV segmentation aux extending v1's 6 channels (3 polylines + 3 agents) with drivable-area / walkway / stop-line. **Subsumes the proposed `dadense` lever** — drop `dadense` as a standalone config and roll drivable area into dseg v2 instead.
- **(Optional) DINOv2 init** — stronger SSL prior on the R50 backbone, in flight on Trillium.

Working hypothesis: stage 1 should be designed for image-feature shaping. Perception is one effective supervisor among several (alongside dense aux2d, dseg, depth, occupancy).

### Stage 2 — perception-free planner ("minS2")

Stage 2 is *only* a planner fine-tune on the stage-1 backbone. Perception heads are either frozen or off. Perception tokens never enter the planner at inference.

Mechanisms, in order of how much each carried:

1. **K/V-off planner** — perception → planner cross-attention (`gnn` for det, `cross_gnn` for map) nulled. No perception tokens flow into the planner at inference.
2. **Planner deformable-to-image (`planpredtrajdeformmm`)** — planner reads image features directly via deformable cross-attn at predicted BEV waypoints. Main L2 driver (0.636 → 0.522).
3. **All-waypoint sampling (`planwp`)** — deformable samples at every trajectory waypoint, not just endpoint.
4. **Last-stage ego instfeat (`planifls`)** — final-stage ego instance feature replaced by image-features deformable readout. Load-bearing; without it `planwp` regresses.
5. **Decoder6** — 6 planning-decoder stages instead of 3.
6. **Ego-status injection (`egostatus`)** — MLP-encode current-frame ego state `[ax, ay, vx, vy, heading]` broadcast-added to the plan mode query at init and each refine. **Single biggest L2 lever post-K/V-off (−0.13).**
7. **Eval-aligned learned rescore (`evalmatchmode`)** — per-mode aggregated BCE with smooth-max(τ=5) over matched anchors at training time, matching inference `any(anchor)` reduction. Recovers the CR signal hard-rescore loses when det K/V is removed.
8. **Image-feature conflict head (Stream B1)** — collision check samples *image features* at BEV query points instead of reading per-agent tokens. Removes the conflict head's dependency on the detection forward pass.
9. **Ego-only planner (Stream C)** — agent slots dropped from `instance_feature`; planner runs ego-only.
10. **Frozen backbone + perception heads (Stream E)** — `lr_mult=0` on backbone / neck / det / map at stage 2; only motion + plan train.

Combined → **minS2**: stage-2 trains only the planner head, on a frozen stage-1 backbone, reading image features directly, with no perception tokens or agent tokens in the planner. Perception heads exist purely as eval-time hooks for NDS / mAP reporting.

### Inference path

```
multi-camera images
        │
        ▼
   ResNet-50 backbone   (stage-1-pretrained for planning-relevant features)
        │
        ▼
       FPN  ─ planning-relevant image features
        │
        ▼
   Planning Decoder      (reads image features directly)
   ─ ego query init from temporal queue + ego state
   ─ 6 stages × [temp_gnn, deformable_to_image, ffn, refine]
   ─ NO perception K/V cross-attention
   ─ all-waypoint deformable + laststage instfeat
        │
        ▼
   Multi-mode trajectory + per-mode confidence
        │
        ▼
   Image-feature conflict head + learned rescore
```

### Training path

```
                       ┌─ det head     ─ aux loss (gradient → backbone)
multi-camera images ─► backbone + FPN ─┤─ map head     ─ aux loss
                                       ├─ motion head  ─ aux loss
                                       ├─ aux2d depth  ─ dense aux loss
                                       ├─ dseg v2      ─ dense BEV-seg aux loss
                                       └─ planning head ─ primary loss
```

Perception heads are auxiliary supervisors. They are not on the inference forward path; they exist for backbone shaping and for eval-reporting (NDS / mAP).

---

## Related works comparison

Six representative camera-based end-to-end planners, characterized by what enters the planner at inference and how perception is supervised at training. Verified by reading each repo's main forward pass.

| Method | Features | Plan ← feats | Plan ← queries | Training supervisors | Inference perception? | Coupling |
|---|---|---|---|---|---|---|
| UniAD [Hu '23] | BEV | ✓ | ✗ (motion via concat-MLP) | det+track+map+motion+occ | ✓ | Sequential |
| VAD / VADv2 [Jiang '23] | BEV | ✗ | ✓ motion + map | det+map+motion | ✓ | Sequential |
| SparseDrive [Sun '24] | Camera | ✗ | ✓ det + map + motion (shared) | det+map+motion | ✓ | Sequential |
| DiffusionDrive [Liao '24] | BEV (cam+LiDAR) † | ✓ | ✓ agent | det+BEV-seg | ✓ | Sequential |
| DriveTransformer [Jia '25] | Camera | ✓ | ✓ agent+map (joint self-attn) | det+map+motion | ✓ | Parallel |
| SSR [Li '25] | BEV (token bottleneck) | ✓ | ✗ | BEV world model | ✗ | Direct |
| LAW [Li '25] | Camera (per-view token bottleneck) | ✓ | ✗ | latent world model | ✗ | Direct |
| **Ours** | **Camera** | ✓ | ✗ | det+map+motion + dense aux | ✗ | **Direct** |

† DiffusionDrive's TransFuser backbone fuses camera + LiDAR; not strictly camera-only.

**Taxonomy:**
- **Sequential** — perception runs first, planning consumes its outputs as concat features, K/V, or shared queries.
- **Parallel** — all task queries co-decode in the same transformer, mixing every layer.
- **Direct** — planner reads features only; no perception/agent/map queries enter the planning decoder.

The bold ✗ in *Inference perception?* is the contribution claim. SSR, LAW, and ours are the only three Direct entries. SSR and LAW share a pattern — **learned token compression + world-model supervision** — while ours is **no compression + standard perception supervisors**. This is the cleanest single-axis differentiation and the strongest narrative angle of the paper.

### Per-method one-liner contrasts

- **UniAD** — Where UniAD cascades motion and occupancy predictions into the planner via concat-MLP fusion before BEV cross-attention, ours removes that cascade entirely; the planner reads image features without any perception or prediction tokens.
- **VAD** — Where VAD chains the planner through dedicated agent and map cross-attention decoders, ours skips both; image features alone carry the agent and map information learned during training.
- **SparseDrive** — Where SparseDrive feeds detection and map tokens as K/V into a joint motion+planning decoder, ours nulls those K/V channels and runs the planner ego-only; perception heads become pure training-time supervisors.
- **DiffusionDrive** — Where DiffusionDrive cross-attends every diffusion step to BEV features and agent queries, ours samples camera features directly at trajectory waypoints; no diffusion loop, no agent-query coupling, camera-only.
- **DriveTransformer** — Where DriveTransformer co-decodes agent, map, and ego queries through joint self-attention every layer, ours decodes the planner alone; no perception or map queries enter the planning transformer.
- **SSR** — Where SSR funnels BEV through a learned 16-token bottleneck, our planner samples multi-view camera features directly at trajectory waypoints; no BEV lift, no compression.
- **LAW** — Where LAW compresses each camera view through learnable view queries and supervises with a latent world model, ours samples camera features directly at trajectory waypoints and supervises with standard perception decoders; no compression, no world model.

---

## Open levers (best-case finishing moves)

- **B1.5 / B1.6 / B1.7 conflict-sampler variants** — close the *last* detection-forward dependency at inference (the current image-feature conflict head still consults det anchor BEV positions for query placement). If a variant ties on both L2 and CR, the architecture is **fully detection/map-free at inference**. B1.6/1.7 already tie L2; CR ~0.025 pp elevated.
- **Apollo joint_nodetach stage-1** (`detach_perception=False`) — last untested stage-1 lever. Either way it sharpens the "stage-1 = shaping, stage-2 = planner-only" framing.
- **dseg v2** — drivable-area + walkway + stop-line channels added to dense seg head; replaces both v1 and the standalone `dadense` lever.
- **NavSim cross-dataset validation** — second dataset confirms the result isn't nuScenes-specific.

## Risks

- B1.x CR doesn't fully close → claim becomes "near-perception-free" rather than "perception-free" at inference. Still a structural break.
- NavSim transfer fails → claim restricted to nuScenes; reviewers will flag.
- Single-seed minS2 result (3397346) needs at least one repro before final write-up.

## Working thesis

**Sparse scene representations are interface-redundant but supervision-essential for planning.** Detection and map tokens carry almost no information into the planner at inference, but their *training signal* shapes the image features the planner does read. Conflating the two roles is structural in current end-to-end stacks (UniAD, VAD, SparseDrive, ParaDrive). Separating them gives a perception-free inference path with no quality cost — and frees stage-1 to use whatever supervision shapes image features best.
