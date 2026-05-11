import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from badish import dataset, loss, model  # noqa: F401 - imported for registry side effects
from badish.utils.registry import DatasetRegistry, LossRegistry, ModelRegistry, OptimRegistry
from badish.utils.utils import set_seeds


def train_one_epoch(model, dataloader, optimizer, criterion, epoch, writer, cfg):
    model.train()
    device = cfg.trainer.device
    running_loss = 0.0

    for idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        outputs = model(X)

        if outputs.ndim != y.ndim:
            outputs = outputs.squeeze()
            y = y.squeeze()

        if outputs.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=outputs.shape[-2:], mode="nearest")

        loss = criterion(outputs, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step = epoch * len(dataloader) + idx
        if global_step % cfg.trainer.log_interval == 0:
            writer.add_scalar("Train/Loss", loss.item(), global_step)

    epoch_loss = running_loss / len(dataloader)
    logging.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)


def train_one_epoch_pu_seg(model, dataloader, optimizer, criterion, epoch, writer, cfg):
    """Pixel-level positive-unlabeled training for dense feature maps."""
    model.train()
    device = cfg.trainer.device
    running_loss = 0.0
    steps = 0

    for idx, (X, M) in enumerate(dataloader):
        X, M = X.to(device), M.to(device)
        logits = model(X)

        if logits.shape[-2:] != M.shape[-2:]:
            M = F.interpolate(M, size=logits.shape[-2:], mode="nearest")

        logits_flat = logits.reshape(-1)
        mask_flat = M.reshape(-1)
        pos_mask = mask_flat > 0.5
        unl_mask = ~pos_mask

        logits_p = logits_flat[pos_mask]
        logits_u = logits_flat[unl_mask]
        if logits_p.numel() == 0 or logits_u.numel() == 0:
            continue

        loss, stats = criterion(logits_p, logits_u)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        steps += 1
        global_step = epoch * len(dataloader) + idx
        if global_step % cfg.trainer.log_interval == 0:
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            with torch.no_grad():
                writer.add_scalar("Train/Prob_Pos", torch.sigmoid(logits_p).mean().item(), global_step)
                writer.add_scalar("Train/Prob_Unl", torch.sigmoid(logits_u).mean().item(), global_step)
            writer.add_scalar("Train/Num_Pos_Pixels", logits_p.numel(), global_step)
            writer.add_scalar("Train/Num_Unl_Pixels", logits_u.numel(), global_step)
            for k, v in stats.items():
                val = float(v.detach().item() if hasattr(v, "detach") else float(v))
                writer.add_scalar(f"Train/{k}", val, global_step)

    epoch_loss = running_loss / max(steps, 1)
    logging.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)


@hydra.main(config_path="config", version_base=None)
def main(cfg: DictConfig):
    log_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, log_dir / "config.yaml")

    logging.info(f"Training started with cfg: {cfg}")
    writer = SummaryWriter(log_dir=log_dir)
    set_seeds(cfg.trainer.seed)

    trainset = DatasetRegistry.get(cfg.traindata.key, **cfg.traindata.params)
    train_loader = DataLoader(
        trainset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
    )

    net = ModelRegistry.get(cfg.model.key, **cfg.model.params).to(cfg.trainer.device)
    optimizer = OptimRegistry.get(cfg.optim.key, net.parameters(), **cfg.optim.params)
    criterion = LossRegistry.get(cfg.loss.key, **cfg.loss.params)

    for epoch in range(cfg.trainer.epochs):
        if cfg.loss.key.startswith("nnpu"):
            train_one_epoch_pu_seg(net, train_loader, optimizer, criterion, epoch, writer, cfg)
        else:
            train_one_epoch(net, train_loader, optimizer, criterion, epoch, writer, cfg)

        if (epoch + 1) % cfg.trainer.ckpt_interval == 0:
            torch.save(net.state_dict(), log_dir / f"model_epoch{epoch + 1}.pth")

    writer.close()
    logging.info("Training completed.")


if __name__ == "__main__":
    main()
