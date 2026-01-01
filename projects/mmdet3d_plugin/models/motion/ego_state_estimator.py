import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class EgoStateEstimator(nn.Module):
    """Predicts current 9-D ego_status from history (and optionally visual features).

    variant='lite': MLP on flattened last-K ego_status only.
    variant='full': history-MLP (raw + deltas) + visual-MLP fused as residual on
        constant-acceleration extrapolation prior.
    """

    def __init__(
        self,
        embed_dims=256,
        ego_status_dim=9,
        history_K=4,
        variant='lite',
        hidden_dim=128,
        use_constant_accel_prior=True,
        aux_loss_weight=0.1,
    ):
        super().__init__()
        assert variant in ('lite', 'full'), variant
        assert history_K >= 1, history_K
        self.variant = variant
        self.history_K = int(history_K)
        self.ego_status_dim = int(ego_status_dim)
        self.embed_dims = int(embed_dims)
        self.use_constant_accel_prior = (
            bool(use_constant_accel_prior) and variant == 'full' and history_K >= 2
        )
        self.aux_loss_weight = float(aux_loss_weight)

        if variant == 'lite':
            in_dim = self.history_K * self.ego_status_dim
            self.history_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.ego_status_dim),
            )
        else:
            history_in = self.history_K * self.ego_status_dim
            delta_in = max(self.history_K - 1, 0) * self.ego_status_dim
            self.history_mlp = nn.Sequential(
                nn.Linear(history_in + delta_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.visual_mlp = nn.Sequential(
                nn.Linear(self.embed_dims, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.ego_status_dim),
            )
            # Initialise residual head's last layer to ~zero so the network
            # starts from the constant-accel prior and learns corrections on
            # top, rather than diverging at step 0.
            nn.init.zeros_(self.fusion_mlp[-1].weight)
            nn.init.zeros_(self.fusion_mlp[-1].bias)

    def forward(self, ego_status_history, ego_feature=None):
        """
        ego_status_history: (bs, K, D) — oldest→newest. Padded with zeros when
            queue is short or when sequence-start mask invalidated it upstream.
        ego_feature: (bs, embed_dims) — required when variant='full'.

        Returns: (bs, D) predicted current-frame ego_status.
        """
        bs = ego_status_history.shape[0]
        if self.variant == 'lite':
            return self.history_mlp(ego_status_history.reshape(bs, -1))

        if ego_feature is None:
            raise ValueError("variant='full' requires ego_feature")
        if ego_feature.dim() == 3 and ego_feature.shape[1] == 1:
            ego_feature = ego_feature.squeeze(1)

        # Constant-acceleration prior: y_{t-1} + (y_{t-1} - y_{t-2}).
        y_t1 = ego_status_history[:, -1]
        if self.history_K >= 2:
            delta = y_t1 - ego_status_history[:, -2]
        else:
            delta = torch.zeros_like(y_t1)
        prior = y_t1 + delta if self.use_constant_accel_prior else y_t1

        # History branch: raw + inter-frame deltas.
        if self.history_K >= 2:
            deltas = ego_status_history[:, 1:] - ego_status_history[:, :-1]
            hist_in = torch.cat(
                [ego_status_history.reshape(bs, -1), deltas.reshape(bs, -1)],
                dim=-1,
            )
        else:
            hist_in = ego_status_history.reshape(bs, -1)
        h_hist = self.history_mlp(hist_in)
        h_vis = self.visual_mlp(ego_feature)
        residual = self.fusion_mlp(torch.cat([h_hist, h_vis], dim=-1))
        return prior + residual

    def loss(self, predicted, target):
        return self.aux_loss_weight * F.l1_loss(predicted, target)
