import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence

def _to_01(x: torch.Tensor) -> torch.Tensor:
    # assumes input in [-1,1] or already [0,1]
    if x.min() < 0:
        x = (x + 1) / 2
    return torch.clamp(x, 0, 1)

def _norm(v: torch.Tensor) -> np.ndarray:
    vmin, vmax = v.min(), v.max()
    if (vmax - vmin) < 1e-12:
        return torch.zeros_like(v).cpu().numpy()
    return ((v - vmin) / (vmax - vmin)).cpu().numpy()

def visualize_vqvae_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: Optional[str] = None,
    channels: Optional[Sequence[int]] = (0,),
    cmap: str = "gray",
    show_error: bool = True,
    title: Optional[str] = None,
    dpi: int = 130,
    show: bool = False,
):
    """
    original, reconstructed: shape [B, C, H, W]
    channels: danh sách kênh để vẽ (ví dụ (0,1,2,3))
    """
    assert original.shape == reconstructed.shape, "Shape mismatch."

    original = _to_01(original.detach())
    reconstructed = _to_01(reconstructed.detach())

    with torch.no_grad():
        mse = torch.mean((reconstructed - original) ** 2).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-12))

    b = 0  # lấy mẫu đầu tiên batch
    chs = channels if channels is not None else range(original.shape[1])
    n_cols = 3 if show_error else 2
    n_rows = len(chs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r, cidx in enumerate(chs):
        ax_orig = axes[r, 0]
        ax_recon = axes[r, 1]

        img_o = original[b, cidx].cpu().numpy()
        img_r = reconstructed[b, cidx].cpu().numpy()

        ax_orig.imshow(img_o, cmap=cmap)
        ax_orig.set_title(f"Orig c{cidx}")
        ax_orig.axis("off")

        ax_recon.imshow(img_r, cmap=cmap)
        ax_recon.set_title(f"Recon c{cidx}")
        ax_recon.axis("off")

        if show_error:
            diff = img_o - img_r
            diff_abs = np.abs(diff)
            ax_err = axes[r, 2]
            im = ax_err.imshow(_norm(torch.from_numpy(diff_abs)), cmap="plasma")
            ax_err.set_title(f"Err |c{cidx}|")
            ax_err.axis("off")
            fig.colorbar(im, ax=ax_err, fraction=0.046, pad=0.04)

    suptitle = title or f"Reconstruction  MSE={mse:.4f}  PSNR={psnr:.2f}dB"
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return {"mse": mse, "psnr": psnr}

# Ví dụ dùng trong loop:
# stats = visualize_vqvae_reconstruction(image, reconstructed, save_path=f"/out/recon_{i:04d}.png", channels=(0,1,2,3))