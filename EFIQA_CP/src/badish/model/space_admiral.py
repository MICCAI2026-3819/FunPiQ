import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from torch import nn


class SpaceAdmiral(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None, "Model not defined."
        return self.model(x)

    @torch.inference_mode()
    def visualize(self, Is: list[np.ndarray], Xs: torch.Tensor, Ms: torch.Tensor, img_cols: int = 3) -> Figure:
        N = len(Is)
        assert N == Xs.shape[0] == Ms.shape[0], "Mismatched input lengths."
        assert self.model is not None, "Model not defined."

        self.model.eval()
        Xs = Xs.permute(0, 3, 1, 2).contiguous()
        preds = torch.sigmoid(self.model(Xs)).squeeze().cpu().numpy()
        Ms = Ms.cpu().numpy()
        Is_np = [i.detach().cpu().numpy() if hasattr(i, "detach") else np.asarray(i) for i in Is]

        cols = img_cols * 3
        rows = N // img_cols + (1 if N % img_cols != 0 else 0)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=100, squeeze=False)

        for r in range(rows):
            for c in range(img_cols):
                i = r * img_cols + c
                if i >= N:
                    for subc in range(3):
                        axes[r, c * 3 + subc].axis("off")
                    continue

                ax = axes[r, c * 3]
                ax.imshow(Is_np[i])
                ax.set_title(f"Image #{i}")
                ax.axis("off")

                ax = axes[r, c * 3 + 1]
                ax.imshow(Ms[i], cmap="gray", vmin=0, vmax=1 if Ms[i].max() <= 1 else Ms[i].max())
                ax.set_title("Mask")
                ax.axis("off")

                ax = axes[r, c * 3 + 2]
                ax.imshow(preds[i], cmap="coolwarm", vmin=0, vmax=1)
                ax.set_title("Prediction")
                ax.axis("off")

        sm = ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, location="right")
        return fig
