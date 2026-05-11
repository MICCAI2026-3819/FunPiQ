import torch
from torch.nn import functional as F

from badish.utils.registry import LossRegistry


@LossRegistry.register("nnpu")
class NNPULoss:
    """Non-negative positive-unlabeled loss for logits."""

    def __init__(self, pi: float, beta: float = 0.0, pos_weight: float = 1.0):
        assert 0 <= pi < 1, "pi must be in [0, 1)"
        self.pi = pi
        self.beta = beta
        self.pos_weight = pos_weight

    def __call__(self, logits_p, logits_u):
        tP = logits_p.reshape(-1)
        tU = logits_u.reshape(-1)

        l_posP = F.softplus(-tP)
        l_negP = F.softplus(tP)
        l_negU = F.softplus(tU)

        pos_term = self.pos_weight * self.pi * l_posP.mean()
        neg_raw = l_negU.mean() - self.pi * l_negP.mean()
        neg_clamped = torch.clamp(neg_raw, min=self.beta)
        loss = pos_term + neg_clamped

        stats = {
            "nnPU_Pos": pos_term.detach(),
            "nnPU_Neg_raw": neg_raw.detach(),
            "nnPU_Neg_clamped": neg_clamped.detach(),
        }
        return loss, stats
