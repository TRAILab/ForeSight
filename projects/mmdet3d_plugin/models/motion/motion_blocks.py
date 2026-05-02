import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        num_driving_cmds=3,
        with_da_head=False,
        with_conflict_head=False,
        plan_mode_time_queries=False,
        plan_per_bucket_reg=False,
        plan_bucket_split=None,
    ):
        super(MotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_driving_cmds = num_driving_cmds
        self.plan_mode_time_queries = plan_mode_time_queries
        # Variant 4: per-bucket plan_reg branch on speedstrat anchors. Modes
        # 0..plan_bucket_split-1 use a "low-speed" reg MLP and the remainder
        # use a "high-speed" reg MLP. Disabled by default; baseline keeps the
        # single shared plan_reg_branch.
        self.plan_per_bucket_reg = bool(plan_per_bucket_reg)
        if self.plan_per_bucket_reg:
            assert plan_bucket_split is not None and 0 < plan_bucket_split < ego_fut_mode, (
                f"plan_per_bucket_reg=True requires 0 < plan_bucket_split < "
                f"ego_fut_mode={ego_fut_mode}; got plan_bucket_split={plan_bucket_split}"
            )
            self.plan_bucket_split = int(plan_bucket_split)

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        # Time-queries mode: each query produces a single (x, y) waypoint.
        # Default: each query produces all ego_fut_ts waypoints at once.
        plan_reg_out = 2 if plan_mode_time_queries else ego_fut_ts * 2
        if self.plan_per_bucket_reg:
            self.plan_reg_branch_low = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, plan_reg_out),
            )
            self.plan_reg_branch_high = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, plan_reg_out),
            )
            self.plan_reg_branch = None
        else:
            self.plan_reg_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, plan_reg_out),
            )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

        self.with_da_head = with_da_head
        if with_da_head:
            # Per-mode per-waypoint drivable-area compliance logit.
            self.plan_da_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, ego_fut_ts),
            )

        self.with_conflict_head = with_conflict_head
        if with_conflict_head:
            # Pairwise (plan_query mode m, agent_feature j) -> conflict logit.
            self.plan_conflict_branch = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, 1),
            )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
        if self.with_da_head:
            nn.init.constant_(self.plan_da_branch[-1].bias, 0.0)
        if self.with_conflict_head:
            # Conflicts are rare — bias toward "no conflict" matches the prior.
            nn.init.constant_(self.plan_conflict_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
        agent_features=None,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        if self.plan_mode_time_queries:
            # plan_query: (bs, 1, M_total*T, D) -> per-(mode, ts) reshape.
            M_total = self.num_driving_cmds * self.ego_fut_mode
            T = self.ego_fut_ts
            plan_query_mt = plan_query.reshape(bs, 1, M_total, T, self.embed_dims)
            # Per-(mode, ts) (x, y).
            assert not self.plan_per_bucket_reg, (
                "plan_per_bucket_reg is not supported with plan_mode_time_queries"
            )
            plan_reg = self.plan_reg_branch(plan_query_mt).reshape(bs, 1, M_total, T, 2)
            # Mode classification: collapse over time via mean before MLP.
            plan_query_mode = plan_query_mt.mean(dim=3)  # (bs, 1, M_total, D)
            plan_cls = self.plan_cls_branch(plan_query_mode).squeeze(-1)
        else:
            plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
            if self.plan_per_bucket_reg:
                # plan_query: (bs, 1, num_modes, D) where
                # num_modes = num_driving_cmds * ego_fut_mode. Within each cmd,
                # modes [0:split) hit the low-bucket reg MLP and [split:M) hit
                # the high-bucket MLP. Reshape so the bucket boundary is
                # contiguous, run the two MLPs, restore order.
                num_cmd = self.num_driving_cmds
                M = self.ego_fut_mode
                split = self.plan_bucket_split
                pq = plan_query.reshape(bs, 1, num_cmd, M, self.embed_dims)
                low = self.plan_reg_branch_low(pq[..., :split, :])
                high = self.plan_reg_branch_high(pq[..., split:, :])
                plan_reg = torch.cat([low, high], dim=-2).reshape(
                    bs, 1, num_cmd * M, self.ego_fut_ts, 2
                )
            else:
                plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, self.num_driving_cmds * self.ego_fut_mode, self.ego_fut_ts, 2)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)

        plan_da = None
        if self.with_da_head:
            # plan_query: (bs, 1, M, D) -> (bs, 1, M, ego_fut_ts) per-waypoint logit.
            plan_da = self.plan_da_branch(plan_query)

        plan_conflict = None
        if self.with_conflict_head and agent_features is not None:
            # plan_query: (bs, 1, M, D); agent_features: (bs, num_anchor, D).
            M = plan_query.shape[2]
            plan_q_flat = plan_query.squeeze(1)  # (bs, M, D)
            agents_exp = agent_features.unsqueeze(2).expand(-1, -1, M, -1)
            plan_exp = plan_q_flat.unsqueeze(1).expand(-1, agent_features.shape[1], -1, -1)
            pair = torch.cat([agents_exp, plan_exp], dim=-1)
            plan_conflict = self.plan_conflict_branch(pair).squeeze(-1)  # (bs, num_anchor, M)

        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status, plan_da, plan_conflict