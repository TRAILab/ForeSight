# Auxiliary 2D Supervision for SparseDrive

## Intro

This report documents an experiment adding auxiliary 2D supervision to `SparseDrive` in order to test whether image-space supervision can improve the quality of learned camera features for 3D detection. The motivation comes from `StreamPETR`, which uses an auxiliary 2D branch during training in addition to its main 3D detection objective.

The goal of this experiment is not to change the core `SparseDrive` detection decoder. Instead, the change isolates the effect of adding a train-time 2D supervision branch while preserving the existing 3D detection pathway as much as possible.

## Method

The experiment adds a train-only auxiliary 2D head to `SparseDrive`. The branch operates on raw image-neck features before deformable feature aggregation formatting and predicts:

- per-location class scores
- 2D bounding boxes
- 2D centers
- centerness

The auxiliary losses follow the same high-level design used by `StreamPETR`:

- `QualityFocalLoss` for classification
- `L1Loss` for 2D box regression
- `GIoULoss` for 2D box overlap
- `L1Loss` for 2D center regression
- `GaussianFocalLoss` for centerness

Because `ForeSight` does not currently store 2D annotations for this setup in the serialized info files, the 2D targets are generated online in the training pipeline. For each 3D ground-truth box and each camera:

- the 3D box corners are projected into the image using the augmented `lidar2img`
- the convex hull of the projected visible corners is intersected with the image canvas
- the resulting clipped 2D box is used as the 2D bbox target
- the projected 3D box center provides the `centers2d` target
- the projected center depth provides the `depths` target

This online target generation was updated to match the `StreamPETR` box construction more closely by using convex-hull/image-canvas intersection rather than a simple min/max over visible corners.

The following experiment configs were added:

- `projects/configs/sparsedrive_r50_stage1_8gpu_noflash_nomap_aux2d.py`
- `projects/configs/sparsedrive_r50_stage1_4gpu_aux2d.py`

These configs wrap their corresponding baselines and enable the aux-2D branch with the additional training targets.

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `dgx` | `sparsedrive_r50_stage1_4gpu_aux2d.py` | `3589` | `RUNNING` |

## Discussion

## Future Work

- Move 2D target generation offline into the dataset conversion pipeline so the setup matches `StreamPETR` more closely.
- Compare online augmented 2D targets against offline precomputed 2D targets to isolate any effect from augmentation-time reprojection.
- Run ablations on aux-2D loss weights to determine whether the branch is helping feature learning or competing with the main 3D objective.
- Test the aux-2D branch both with and without auxiliary depth supervision to measure whether the two auxiliary signals are complementary.
- Evaluate whether aux-2D supervision interacts with denoising training in `SparseDrive`.
