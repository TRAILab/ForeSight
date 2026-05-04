"""Custom collate for SparseDrive's variable-length per-sample GT.

Lightning's default collate (`torch.utils.data.default_collate`) calls
`torch.stack` over batch values, which fails when per-sample tensor shapes
differ (e.g. `gt_bboxes_3d` is `(N_i, 9)` with `N_i` varying per scene).

Strategy: try `default_collate` first; if it raises a shape-related
RuntimeError on a key, fall back to keeping that key as a Python list of
the per-sample values. The SparseDrive head's `loss(...)` accepts
list-shaped per-batch GT for det/map/motion already (see
`SparseDriveHead.loss` -> `det_head.loss` which iterates the list).

Keys that were always stackable (img, ego_status, T_global, ...) keep
their tensor batching; keys that aren't stackable become lists. No schema
change required on the model side.
"""

from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


def _try_stack(values: Sequence[Any]) -> Any:
    """Stack a list of per-sample values; fall back to a Python list when
    they're variable-length tensors that default_collate refuses."""
    if not values:
        return values
    first = values[0]
    if isinstance(first, torch.Tensor):
        # All tensors with identical shape → stack; otherwise return as list
        # (the head's loss path accepts list[Tensor] for variable-N targets).
        if all(isinstance(v, torch.Tensor) and v.shape == first.shape for v in values):
            return torch.stack(list(values), dim=0)
        return list(values)
    if isinstance(first, dict):
        # Recurse on sub-dicts (rare in our schema but kept for safety).
        return {k: _try_stack([v[k] for v in values]) for k in first.keys()}
    if isinstance(first, list):
        # Already a list (e.g. per-sample per-instance lists). Pass through.
        return list(values)
    # Numbers / strings / None — let default_collate handle them.
    return default_collate(list(values))


def sparsedrive_collate(
    batch: List[Any],
) -> Any:
    """Top-level collate. Each `batch` element is `(features_dict, targets_dict)`
    from the navsim Dataset; we collate the two dicts independently."""
    if not batch:
        return batch
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2:
        feats = [b[0] for b in batch]
        targs = [b[1] for b in batch]
        return _collate_dict(feats), _collate_dict(targs)
    if isinstance(batch[0], dict):
        return _collate_dict(batch)
    return default_collate(batch)


def _collate_dict(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack per-key, falling back to lists for variable-length tensors."""
    keys = batch[0].keys()
    return {k: _try_stack([b[k] for b in batch]) for k in keys}
