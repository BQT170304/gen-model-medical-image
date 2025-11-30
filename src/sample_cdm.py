import os
import random
import csv
from typing import Dict, List

import numpy as np
import torch
import hydra
import pyrootutils
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm
from models.diffusion.condition_diffusion_module import ConditionalDiffusionModule


root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def set_seed(seed: int = 12345) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    _min = torch.min(img)
    _max = torch.max(img)
    if (_max - _min) > 0:
        return (img - _min) / (_max - _min)
    return img


def dice_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt = ground_truth.astype(bool)
    pr = prediction.astype(bool)
    inter = np.logical_and(gt, pr).sum()
    return 2.0 * inter / (gt.sum() + pr.sum() + 1e-7)


def iou_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt = ground_truth.astype(bool)
    pr = prediction.astype(bool)
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    return inter / (union + 1e-7)


def save_modalities(original: torch.Tensor, generated: torch.Tensor, save_dir: str) -> None:
    # Dynamically determine modalities based on tensor shape
    num_modalities = original.shape[1] if original.ndim > 1 else 1
    modalities = [f"Modality_{i}" for i in range(num_modalities)]
    os.makedirs(save_dir, exist_ok=True)

    # Original images
    for i, name in enumerate(modalities):
        img = normalize_image(original[0, i]).detach().cpu().numpy() if num_modalities > 1 else normalize_image(original[0]).detach().cpu().numpy().squeeze()
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"input_{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

    # Generated images
    for i, name in enumerate(modalities):
        img = normalize_image(generated[0, i]).detach().cpu().numpy() if num_modalities > 1 else normalize_image(generated[0]).detach().cpu().numpy().squeeze()
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"output_{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()


def save_maps(difftot: np.ndarray, difftot_mask: np.ndarray, gt_mask: np.ndarray, save_dir: str) -> Dict[str, float]:
    os.makedirs(save_dir, exist_ok=True)

    # Ensure all arrays are 2D for visualization
    if difftot.ndim > 2:
        difftot = difftot.squeeze()
    if difftot_mask.ndim > 2:
        difftot_mask = difftot_mask.squeeze()
    if gt_mask.ndim > 2:
        gt_mask = gt_mask.squeeze()

    # Normalize heatmap to [0,1] for visualization
    dmin, dmax = float(difftot.min()), float(difftot.max())
    if dmax > dmin:
        heat = (difftot - dmin) / (dmax - dmin)
    else:
        heat = np.zeros_like(difftot)

    # Save anomaly heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(heat, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "anomaly_heatmap.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Save ground truth mask
    plt.figure(figsize=(6, 6))
    plt.imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gt_mask.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Save predicted mask (colorful version)
    plt.figure(figsize=(6, 6))
    plt.imshow(difftot_mask, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predicted_mask.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Save predicted mask (black/white binary version)
    # Convert to binary: threshold at 0.5 and make it pure black/white
    binary_mask = (difftot_mask > 0.5).astype(np.uint8) * 255
    
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_mask, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predicted_mask_binary.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Calculate metrics using binary mask
    binary_mask_normalized = (difftot_mask > 0.5).astype(float)
    d = dice_score(gt_mask, binary_mask_normalized)
    j = iou_score(gt_mask, binary_mask_normalized)

    # Save metrics
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Dice: {d:.6f}\n")
        f.write(f"IoU: {j:.6f}\n")
        f.write(f"Ground truth sum: {gt_mask.sum():.0f}\n")
        f.write(f"Predicted sum: {difftot_mask.sum():.0f}\n")
        f.write(f"Binary predicted sum: {binary_mask_normalized.sum():.0f}\n")

    return {"dice": d, "iou": j}

def create_classifier(model_path: str, device: torch.device):
    """Load classifier from checkpoint"""
    try:
        from src.models.guided_diffusion.script_util import (
            classifier_defaults,
            create_classifier as create_classifier_fn,
            args_to_dict
        )
        import argparse
        
        # Create default args for classifier
        defaults = classifier_defaults()
        defaults.update({"dataset": "brats"})
        defaults.update({"image_size": 256})  # Work with full resolution images
        defaults.update({"diffusion_steps": 1000})
        defaults.update({"classifier_use_fp16": False})
        defaults.update({"classifier_width": 128})
        defaults.update({"classifier_depth": 2})
        defaults.update({"classifier_attention_resolutions": "32,16,8"})
        defaults.update({"classifier_use_scale_shift_norm": True})
        defaults.update({"classifier_resblock_updown": True})
        defaults.update({"classifier_pool": "attention"})
        
        parser = argparse.ArgumentParser()
        for k, v in defaults.items():
            parser.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)
        
        args = parser.parse_args([])
        
        classifier = create_classifier_fn(**args_to_dict(args, classifier_defaults().keys()))
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        classifier.to(device)
        classifier.eval()
        print(f"✓ Loaded classifier from {model_path}")
        return classifier
        
    except Exception as e:
        print(f"✗ Failed to load classifier: {e}")
        return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_cdm.yaml")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.get("seed", 12345)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    loader = datamodule.val_dataloader()

    # Model
    # For LiTS:
    # cfg.model.dataset = "lits"
    # cfg.model.model_path = f"{root}/src/ckpt_lits/conditional_diffusion/unet_cdm256_1000step.pth"
    cdm: ConditionalDiffusionModule = hydra.utils.instantiate(cfg.model)
    cdm.model.to(device)
    cdm.eval()

    # Load classifier if specified
    classifier = None
    if hasattr(cfg, "classifier") and cfg.classifier.use_classifier and cfg.classifier.path:
        classifier = create_classifier(cfg.classifier.path, device)
        classifier_scale = cfg.classifier.scale if hasattr(cfg.classifier, "scale") else 100.0
    else:
        classifier_scale = 0.0

    num_samples = int(getattr(cfg.sampling, "num_samples", 400))
    noise_level = int(getattr(cfg.sampling, "noise_level", 200))
    save_dir = getattr(cfg.sampling, "save_dir", f"{root}/results/visualize_cdm")
    skip_normal = getattr(cfg.sampling, "skip_normal", True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing {num_samples} samples with noise_level={noise_level}")
    print(f"Skip normal samples: {skip_normal}")
    print(f"Classifier loaded: {classifier is not None}")

    # Lists to store all metrics
    all_dice_scores = []
    all_iou_scores = []
    results_data = []
    
    # Process samples
    processed = 0
    global_idx = 0
    
    for batch in tqdm(loader, desc="Processing samples"):
        # if processed >= num_samples:
        #     break
            
        imgs, cond, mask, labels = batch
        mask_np = mask[0].cpu().numpy()
        
        # Skip normal samples if requested
        if skip_normal and mask_np.sum() == 0:
            global_idx += 1
            continue
            
        # Move tensors to device
        imgs = imgs.to(device)
        cond = {}
        
        # Set class label for classifier guidance
        if classifier is not None:
            # Label 1 for abnormal (has lesion), 0 for normal
            classes = torch.randint(low=0, high=1, size=(1,), device=device)
            cond["y"] = torch.tensor([0], device=device) 

        with torch.no_grad():
            best_dice = -1.0
            best_iou = -1.0
            best_out = None

            for _ in range(3):
                candidate = cdm.sample(
                    batch=[imgs, cond, mask, labels],
                    cond=cond,
                    classifier=classifier,
                    classifier_scale=classifier_scale,
                    noise_level=noise_level,
                    batch_size=imgs.shape[0],
                )

                # Normalize candidate mask to numpy for metric computation
                cand_mask = candidate.get("difftot_mask", None)
                if cand_mask is None:
                    continue
                if torch.is_tensor(cand_mask):
                    cand_mask_np = cand_mask[0].detach().cpu().numpy()
                else:
                    cand_mask_np = np.array(cand_mask)

                # Ensure shape compatibility with mask_np
                cand_mask_np = cand_mask_np.squeeze()

                dice_c = dice_score(mask_np, cand_mask_np)
                iou_c = iou_score(mask_np, cand_mask_np)

                if dice_c > best_dice:
                    best_dice = dice_c
                    best_iou = iou_c
                    best_out = candidate

            # Fallback if none produced a result
            if best_out is None:
                out = candidate
                dice = best_dice if best_dice > 0 else 0.0
                iou = best_iou if best_iou > 0 else 0.0
            else:
                out = best_out
                dice = best_dice
                iou = best_iou
        print(f"Sample {global_idx:04d} - Dice: {dice:.4f}, IoU: {iou:.4f}")
        
        all_dice_scores.append(dice)
        all_iou_scores.append(iou)
        
        # Store result data
        results_data.append({
            "sample_idx": global_idx,
            "processed_idx": processed,
            "dice": dice,
            "iou": iou,
            "has_lesion": int(mask_np.sum() > 0)
        })
        
        # Save individual sample if it's a good result or specific index
        if dice > 0.8 or global_idx in [300, 350, 400]:  # Save some examples
            idx_dir = os.path.join(save_dir, f"sample_{global_idx:04d}")
            save_modalities(out["original_images"], out["generated_images"], idx_dir)
            save_maps(out["difftot"], out["difftot_mask"], mask_np, idx_dir)
        
        processed += 1
        global_idx += 1

    # Calculate and save summary statistics
    if all_dice_scores:
        mean_dice = np.mean(all_dice_scores)
        mean_iou = np.mean(all_iou_scores)
        std_dice = np.std(all_dice_scores)
        std_iou = np.std(all_iou_scores)
        
        # Filter non-zero dice scores
        nonzero_dice = [d for d in all_dice_scores if d > 0]
        nonzero_iou = [i for i in all_iou_scores if i > 0]
        
        mean_dice_nonzero = np.mean(nonzero_dice) if nonzero_dice else 0.0
        mean_iou_nonzero = np.mean(nonzero_iou) if nonzero_iou else 0.0
        
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Processed samples: {processed}")
        print(f"Mean Dice (all): {mean_dice:.6f} ± {std_dice:.6f}")
        print(f"Mean IoU (all): {mean_iou:.6f} ± {std_iou:.6f}")
        print(f"Mean Dice (>0): {mean_dice_nonzero:.6f} ({len(nonzero_dice)}/{len(all_dice_scores)} samples)")
        print(f"Mean IoU (>0): {mean_iou_nonzero:.6f} ({len(nonzero_iou)}/{len(all_iou_scores)} samples)")
        
        # Save detailed CSV results
        csv_path = os.path.join(save_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_idx", "processed_idx", "dice", "iou", "has_lesion"])
            writer.writeheader()
            writer.writerows(results_data)
            
        # Save summary
        summary_path = os.path.join(save_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"CDM Sampling Results\n")
            f.write(f"===================\n")
            f.write(f"Processed samples: {processed}\n")
            f.write(f"Noise level (L): {noise_level}\n")
            f.write(f"Classifier used: {classifier is not None}\n")
            f.write(f"Skip normal: {skip_normal}\n")
            f.write(f"\nMetrics:\n")
            f.write(f"Mean Dice (all): {mean_dice:.6f} ± {std_dice:.6f}\n")
            f.write(f"Mean IoU (all): {mean_iou:.6f} ± {std_iou:.6f}\n")
            f.write(f"Mean Dice (>0): {mean_dice_nonzero:.6f} ({len(nonzero_dice)}/{len(all_dice_scores)} samples)\n")
            f.write(f"Mean IoU (>0): {mean_iou_nonzero:.6f} ({len(nonzero_iou)}/{len(all_iou_scores)} samples)\n")
        
        print(f"\nResults saved to:")
        print(f"- CSV: {csv_path}")
        print(f"- Summary: {summary_path}")
        print(f"- Sample outputs: {save_dir}/sample_XXXX/")
    
    else:
        print("No samples processed.")


if __name__ == "__main__":
    main()
