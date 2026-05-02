"""SceneQueryDecoder: K learnable BEV anchors refined by image features.

Use-case: provide the planner's conflict head with adaptive scene queries
that don't depend on a detection forward pass. K=50 anchors initialised
from a kmeans of ego-interacting agents are passed through 2 layers of
(deformable image-feature attention → self-attention → FFN → refine MLP),
yielding per-scene refined BEV positions.

Output: dict with `anchors` (B, K, 11) and `features` (B, K, D).
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner import BaseModule
from mmcv.utils import build_from_cfg

from ..attention import gen_sineembed_for_position


class SceneQueryRefineMLP(nn.Module):
    """Outputs Δ(x, y, z) added to the anchor's position dims [0:3]."""

    def __init__(self, embed_dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, query, anchor):
        delta = self.layers(query)
        out = anchor.clone()
        out[..., :3] = anchor[..., :3] + delta
        return out


class SceneQueryLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, dropout, deformable_cfg, ffn_dim):
        super().__init__()
        self.deformable = build_from_cfg(deformable_cfg, ATTENTION)
        self.norm_def = nn.LayerNorm(embed_dims)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=ffn_dim,
            num_fcs=2,
            ffn_drop=dropout,
            act_cfg=dict(type="ReLU", inplace=True),
        )
        self.norm_ffn = nn.LayerNorm(embed_dims)
        self.refine = SceneQueryRefineMLP(embed_dims)

    def forward(self, query, anchor, anchor_embed, feature_maps, metas):
        # deformable: query-q + anchor-pos → image features
        query = self.deformable(query, anchor, anchor_embed, feature_maps, metas)
        query = self.norm_def(query)
        # self-attn among queries
        sa_out, _ = self.self_attn(query, query, query)
        query = self.norm_sa(query + sa_out)
        # FFN
        query = self.norm_ffn(self.ffn(query))
        # Refine anchor (Δxyz). Returns new anchor; caller re-encodes embed.
        anchor = self.refine(query, anchor)
        return query, anchor


@ATTENTION.register_module()
class SceneQueryDecoder(BaseModule):
    """K learnable BEV anchors → 2 refinement layers → refined (anchors, features).

    Args:
        embed_dims: feature dim
        num_queries: K (number of scene queries)
        anchor_path: path to (K, 11) numpy file (init anchors)
        num_layers: refinement depth
        deformable_cfg: dict for the per-layer DeformableFeatureAggregation
        num_heads: self-attn heads
        ffn_dim: FFN inner channels
        dropout: shared dropout
    """

    def __init__(
        self,
        embed_dims,
        num_queries,
        anchor_path,
        deformable_cfg,
        num_layers=2,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        anchors = np.load(anchor_path)
        if anchors.shape[0] < num_queries:
            raise ValueError(
                f"anchor file {anchor_path} has only {anchors.shape[0]} anchors, "
                f"need at least num_queries={num_queries}"
            )
        anchors = anchors[:num_queries]
        assert anchors.shape[-1] == 11, f"expected 11-dim anchors, got {anchors.shape}"
        self.register_buffer(
            "init_anchors", torch.from_numpy(anchors).float(), persistent=False
        )
        self.query_embed = nn.Parameter(torch.zeros(num_queries, embed_dims))
        nn.init.normal_(self.query_embed, std=0.02)
        # gen_sineembed_for_position consumes (x, y) and outputs embed_dims.
        # Light projection on top to mix with the learnable query embed.
        self.anchor_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.layers = nn.ModuleList(
            [
                SceneQueryLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    dropout=dropout,
                    deformable_cfg=deformable_cfg,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def _encode_anchor(self, anchor):
        # gen_sineembed_for_position reads (x, y) and returns (..., embed_dims).
        emb = gen_sineembed_for_position(anchor[..., :2], hidden_dim=self.embed_dims)
        return self.anchor_proj(emb)

    def forward(self, feature_maps, metas):
        # feature_maps and metas come from the parent head's forward.
        bs = feature_maps[0].shape[0] if isinstance(feature_maps, (list, tuple)) else feature_maps.shape[0]
        anchor = self.init_anchors.unsqueeze(0).expand(bs, -1, -1).contiguous()
        query = self.query_embed.unsqueeze(0).expand(bs, -1, -1).contiguous()
        for layer in self.layers:
            anchor_embed = self._encode_anchor(anchor)
            query = query + anchor_embed
            query, anchor = layer(query, anchor, anchor_embed, feature_maps, metas)
        return dict(anchors=anchor, features=query)
