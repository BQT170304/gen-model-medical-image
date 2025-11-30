import os
import torch
import torch.nn.functional as F
import numpy as np
import pyrootutils
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
import lpips

# Setup root
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.vae_module import VAEModule

# ---------------------- Normalization Helpers ----------------------
def normalize_to_range(tensor, target_min=0, target_max=1):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    if tensor_max > tensor_min:
        normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized * (target_max - target_min) + target_min
    return tensor

# LPIPS expects [-1,1] and 3 channels
def prepare_images_for_lpips(images):
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.shape[1] == 4:
        images = images[:, :3]
    images = normalize_to_range(images, -1, 1)
    return images

# MS-SSIM expects [0,1]
def prepare_images_for_msssim(images):
    return normalize_to_range(images, 0, 1)

# ---------------------- Metric Computations ----------------------
def compute_ms_ssim(original, reconstructed):
    original = prepare_images_for_msssim(original)
    reconstructed = prepare_images_for_msssim(reconstructed)
    if original.shape[-1] < 160 or original.shape[-2] < 160:
        original = F.interpolate(original, size=(160, 160), mode='bilinear', align_corners=False)
        reconstructed = F.interpolate(reconstructed, size=(160, 160), mode='bilinear', align_corners=False)
    values = []
    for c in range(original.shape[1]):
        val = ms_ssim(original[:, c:c+1], reconstructed[:, c:c+1], data_range=1.0, size_average=True)
        values.append(val.item())
    return float(np.mean(values))

def compute_lpips(original, reconstructed, lpips_model):
    original = prepare_images_for_lpips(original)
    reconstructed = prepare_images_for_lpips(reconstructed)
    return lpips_model(original, reconstructed).mean().item()

# ---------------------- Visualization ----------------------
def save_reconstruction_visuals(original: torch.Tensor,
                                reconstructed: torch.Tensor,
                                dataset_name: str,
                                batch_idx: int,
                                save_dir: str) -> None:
    try:
        os.makedirs(save_dir, exist_ok=True)
        orig_cpu = original.detach().cpu()
        recon_cpu = reconstructed.detach().cpu()
        if orig_cpu.dim() != 4:
            return
        n_channels = orig_cpu.shape[1]
        vis_channels = min(n_channels, 4)
        fig, axes = plt.subplots(2, vis_channels, figsize=(3 * vis_channels, 6))
        if dataset_name.lower().startswith("brats"):
            modality_names = ["T1", "T1ce", "T2", "FLAIR"][:vis_channels]
        else:
            modality_names = ["CT"] * vis_channels
        for c in range(vis_channels):
            axes[0, c].imshow(normalize_to_range(orig_cpu[0, c]).numpy(), cmap='gray')
            axes[0, c].set_title(f"Orig {modality_names[c]}")
            axes[0, c].axis('off')
            axes[1, c].imshow(normalize_to_range(recon_cpu[0, c]).numpy(), cmap='gray')
            axes[1, c].set_title(f"Recon {modality_names[c]}")
            axes[1, c].axis('off')
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{dataset_name}_reconstruction_batch_{batch_idx:04d}.png")
        plt.savefig(out_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Visualization error (batch {batch_idx}): {e}")

# ---------------------- Evaluation Class ----------------------
class ReconstructionMetrics:
    def __init__(self, device='cuda', lpips_network='alex'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lpips_model = lpips.LPIPS(net=lpips_network).to(self.device)

    def _load_vqvae_weights(self, model, cfg):
        enc_path = cfg.evaluation.get('encoder_path')
        dec_path = cfg.evaluation.get('decoder_path')
        vq_path = cfg.evaluation.get('vq_layer_path')
        if not hasattr(model, 'net'):
            return
        vq_vae = model.net
        try:
            if enc_path and os.path.exists(enc_path):
                vq_vae.encoder.load_state_dict(torch.load(enc_path, map_location=self.device))
            if dec_path and os.path.exists(dec_path):
                vq_vae.decoder.load_state_dict(torch.load(dec_path, map_location=self.device))
            if vq_path and os.path.exists(vq_path):
                vq_vae.vq_layer.load_state_dict(torch.load(vq_path, map_location=self.device))
            print("Loaded VQ-VAE component weights.")
        except Exception as e:
            print(f"Weight loading error: {e}")

    def evaluate_vqvae(self, cfg, dataset_name="brats2020"):
        print(f"Evaluating VQ-VAE on {dataset_name} ...")
        model: VAEModule = hydra.utils.instantiate(cfg.model)
        model.to(self.device)
        model.eval()
        self._load_vqvae_weights(model, cfg)
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()
        val_loader = datamodule.val_dataloader()
        max_batches = cfg.evaluation.get('max_batches', 100)
        if max_batches == -1:
            max_batches = len(val_loader)
        save_dir = cfg.evaluation.get('save_dir', 'results/reconstruction_metrics')
        os.makedirs(save_dir, exist_ok=True)

        ms_ssim_scores, lpips_scores, mse_scores, psnr_scores = [], [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Reconstructing")):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(self.device)
                try:
                    # Forward through VAEModule -> returns reconstruction + loss dict
                    reconstructed, _losses = model(images)
                except Exception as e:
                    print(f"Forward error batch {batch_idx}: {e}")
                    continue

                # Metrics
                mse_val = F.mse_loss(reconstructed, images).item()
                mse_scores.append(mse_val)
                psnr_val = 10 * np.log10(1.0 / (mse_val + 1e-12))
                psnr_scores.append(psnr_val)
                ms_ssim_scores.append(compute_ms_ssim(images, reconstructed))
                lpips_scores.append(compute_lpips(images, reconstructed, self.lpips_model))

                if cfg.evaluation.get('save_visuals', False):
                    max_vis = cfg.evaluation.get('max_visuals', 16)
                    if batch_idx < max_vis:
                        vis_dir = os.path.join(save_dir, 'visuals')
                        save_reconstruction_visuals(images, reconstructed, dataset_name, batch_idx, vis_dir)

                if batch_idx % 50 == 0:
                    print(f"Batch {batch_idx}: MSE={mse_val:.4f} PSNR={psnr_val:.2f} MS-SSIM={ms_ssim_scores[-1]:.4f} LPIPS={lpips_scores[-1]:.4f}")

                if batch_idx >= max_batches - 1:
                    break

        # Aggregate
        def stats(arr):
            return float(np.mean(arr)), float(np.std(arr))
        mean_ms, std_ms = stats(ms_ssim_scores)
        mean_lp, std_lp = stats(lpips_scores)
        mean_mse, std_mse = stats(mse_scores)
        mean_psnr, std_psnr = stats(psnr_scores)

        print("\n" + "="*55)
        print(f"Reconstruction Metrics ({dataset_name.upper()})")
        print("="*55)
        print(f"MSE     : {mean_mse:.6f} ± {std_mse:.6f}")
        print(f"PSNR    : {mean_psnr:.3f} ± {std_psnr:.3f} dB")
        print(f"MS-SSIM : {mean_ms:.4f} ± {std_ms:.4f}")
        print(f"LPIPS   : {mean_lp:.4f} ± {std_lp:.4f}")
        print(f"Samples : {len(ms_ssim_scores)}")
        print("="*55)

        results = {
            'dataset': dataset_name,
            'mse_mean': mean_mse,
            'mse_std': std_mse,
            'psnr_mean': mean_psnr,
            'psnr_std': std_psnr,
            'ms_ssim_mean': mean_ms,
            'ms_ssim_std': std_ms,
            'lpips_mean': mean_lp,
            'lpips_std': std_lp,
            'num_samples': len(ms_ssim_scores),
            'mse_scores': mse_scores,
            'psnr_scores': psnr_scores,
            'ms_ssim_scores': ms_ssim_scores,
            'lpips_scores': lpips_scores,
        }

        results_file = os.path.join(save_dir, f"{dataset_name}_vqvae_metrics.txt")
        with open(results_file, 'w') as f:
            f.write(f"Reconstruction Metrics for {dataset_name.upper()}\n")
            f.write("="*55 + "\n")
            f.write(f"MSE     : {mean_mse:.6f} ± {std_mse:.6f}\n")
            f.write(f"PSNR    : {mean_psnr:.3f} ± {std_psnr:.3f} dB\n")
            f.write(f"MS-SSIM : {mean_ms:.4f} ± {std_ms:.4f}\n")
            f.write(f"LPIPS   : {mean_lp:.4f} ± {std_lp:.4f}\n")
            f.write(f"Samples : {len(ms_ssim_scores)}\n")
            f.write("="*55 + "\n\n")
            for i, (mse_v, psnr_v, ms_v, lp_v) in enumerate(zip(mse_scores, psnr_scores, ms_ssim_scores, lpips_scores)):
                f.write(f"Sample {i:03d}: MSE={mse_v:.6f} PSNR={psnr_v:.2f} MS-SSIM={ms_v:.4f} LPIPS={lp_v:.4f}\n")
        print(f"Results saved: {results_file}")

        self._plot_distributions(ms_ssim_scores, lpips_scores, mse_scores, psnr_scores, dataset_name, save_dir)
        return results

    def _plot_distributions(self, ms_ssim_scores, lpips_scores, mse_scores, psnr_scores, dataset_name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        # Histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0,0].hist(mse_scores, bins=30, color='purple', alpha=0.7, edgecolor='black')
        axes[0,0].set_title('MSE Distribution')
        axes[0,0].axvline(np.mean(mse_scores), color='red', linestyle='--')
        axes[0,1].hist(psnr_scores, bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[0,1].set_title('PSNR Distribution')
        axes[0,1].axvline(np.mean(psnr_scores), color='red', linestyle='--')
        axes[1,0].hist(ms_ssim_scores, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[1,0].set_title('MS-SSIM Distribution')
        axes[1,0].axvline(np.mean(ms_ssim_scores), color='red', linestyle='--')
        axes[1,1].hist(lpips_scores, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[1,1].set_title('LPIPS Distribution')
        axes[1,1].axvline(np.mean(lpips_scores), color='red', linestyle='--')
        for ax in axes.flatten():
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_metric_histograms.png"), dpi=250, bbox_inches='tight')
        plt.close()

        # Scatter MS-SSIM vs LPIPS
        plt.figure(figsize=(7,6))
        plt.scatter(ms_ssim_scores, lpips_scores, s=18, alpha=0.6)
        plt.xlabel('MS-SSIM')
        plt.ylabel('LPIPS')
        plt.title(f'MS-SSIM vs LPIPS - {dataset_name.upper()}')
        corr = np.corrcoef(ms_ssim_scores, lpips_scores)[0,1]
        plt.text(0.05,0.95, f'Corr: {corr:.3f}', transform=plt.gca().transAxes)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_msssim_vs_lpips.png"), dpi=250, bbox_inches='tight')
        plt.close()

# ---------------------- Entry Points ----------------------
def evaluate_brats_vqvae():
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(config_name="eval_brats_vqvae")
    metrics = ReconstructionMetrics(device=cfg.evaluation.device, lpips_network=cfg.evaluation.lpips_network)
    return metrics.evaluate_vqvae(cfg, "brats2020")

def evaluate_lits_vqvae():
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(config_name="eval_lits_vqvae")
    metrics = ReconstructionMetrics(device=cfg.evaluation.device, lpips_network=cfg.evaluation.lpips_network)
    return metrics.evaluate_vqvae(cfg, "lits")

def main():
    print("Starting Reconstruction Metrics Evaluation")
    print("="*60)
    print("1. BraTS2020 VQ-VAE Evaluation")
    try:
        brats_results = evaluate_brats_vqvae()
        print(f"BraTS - MS-SSIM {brats_results['ms_ssim_mean']:.4f} LPIPS {brats_results['lpips_mean']:.4f} PSNR {brats_results['psnr_mean']:.2f}dB")
    except Exception as e:
        print(f"BraTS evaluation failed: {e}")
        brats_results = None
    print("="*60)
    # Uncomment to evaluate LiTS
    # print("2. LiTS VQ-VAE Evaluation")
    # try:
    #     lits_results = evaluate_lits_vqvae()
    #     print(f"LiTS - MS-SSIM {lits_results['ms_ssim_mean']:.4f} LPIPS {lits_results['lpips_mean']:.4f} PSNR {lits_results['psnr_mean']:.2f}dB")
    # except Exception as e:
    #     print(f"LiTS evaluation failed: {e}")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
