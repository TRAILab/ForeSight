import torch
import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class PlanningOnlyRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=6,
        ego_fut_mode=6,
        num_driving_cmds=3,
        with_conflict_head=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_driving_cmds = num_driving_cmds

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

        self.with_conflict_head = with_conflict_head
        if with_conflict_head:
            self.plan_conflict_branch = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, 1),
            )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
        if self.with_conflict_head:
            nn.init.constant_(self.plan_conflict_branch[-1].bias, bias_init)

    def forward(
        self,
        plan_query,
        ego_feature,
        ego_anchor_embed,
        agent_features=None,
    ):
        bs = plan_query.shape[0]
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_reg = self.plan_reg_branch(plan_query).reshape(
            bs, 1, self.num_driving_cmds * self.ego_fut_mode, self.ego_fut_ts, 2
        )
        plan_status = self.plan_status_branch(ego_feature + ego_anchor_embed)

        plan_conflict = None
        if self.with_conflict_head and agent_features is not None:
            M = plan_query.shape[2]
            plan_q_flat = plan_query.squeeze(1)  # (bs, M, D)
            agents_exp = agent_features.unsqueeze(2).expand(-1, -1, M, -1)
            plan_exp = plan_q_flat.unsqueeze(1).expand(
                -1, agent_features.shape[1], -1, -1
            )
            pair = torch.cat([agents_exp, plan_exp], dim=-1)
            plan_conflict = self.plan_conflict_branch(pair).squeeze(-1)

        return plan_cls, plan_reg, plan_status, plan_conflict
