import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRegistry:
    ITEM_TYPE = "item"

    @classmethod
    def register(cls, name: str | None = None, *, overwrite: bool = False):
        def decorator(obj):
            key = (name or obj.__name__).lower()
            if not overwrite and key in cls._registry:
                raise KeyError(f"{cls.ITEM_TYPE} '{key}' already registered.")
            cls._registry[key] = obj
            return obj

        return decorator

    @classmethod
    def get(cls, name: str, *args, **kwargs):
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"{cls.ITEM_TYPE} '{name}' not in registry.")
        return cls._registry[key](*args, **kwargs)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())


class ModelRegistry(BaseRegistry):
    ITEM_TYPE = "Model"
    _registry = {}


class LossRegistry(BaseRegistry):
    ITEM_TYPE = "Loss"
    _registry = {}


class OptimRegistry(BaseRegistry):
    ITEM_TYPE = "Optim"
    _registry = {}


class DatasetRegistry(BaseRegistry):
    ITEM_TYPE = "Dataset"
    _registry = {}


class TrainerRegistry(BaseRegistry):
    ITEM_TYPE = "Trainer"
    _registry = {}


LossRegistry._registry = {
    "bce": nn.BCEWithLogitsLoss,
}


@LossRegistry.register("bce_pos_weight")
class BCEWithLogitsLossPosWeight(nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight: float = 1.0, **kwargs):
        super().__init__(pos_weight=torch.tensor(pos_weight), **kwargs)


@LossRegistry.register("focal_bce")
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


OptimRegistry._registry = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
