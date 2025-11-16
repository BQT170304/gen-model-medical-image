import os
import argparse
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pyrootutils
import hydra
from omegaconf import DictConfig, OmegaConf
from models.diffusion.latent_diffusion_module import LatentDiffusionModule
from pytorch_lightning import LightningDataModule, LightningModule
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc

from models.guided_diffusion import dist_util, logger
from models.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    classifier_defaults,
    create_classifier,
)

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=100,
        dataset='brats'
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def normalize_image(img):
    """Normalize image to [0, 1] range"""
    _min = img.min()
    _max = img.max()
    if _max > _min:
        return (img - _min) / (_max - _min)
    return img

def create_anomaly_heatmap(original, reconstructed, method='l1'):
    """Create anomaly heatmap from original and reconstructed images"""
    if method == 'mse':
        diff = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
    elif method == 'l1':
        diff = torch.mean(torch.abs(original - reconstructed), dim=1, keepdim=True)
    else:
        diff = torch.mean(torch.abs(original - reconstructed), dim=1, keepdim=True)
    
    return diff.squeeze().cpu().numpy()

def dice_score(ground_truth, prediction):
    """Calculate the Dice coefficient between two binary masks."""
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    dice = 2.0 * intersection / (ground_truth.sum() + prediction.sum() + 1e-7)
    return dice

def iou_score(ground_truth, prediction):
    """Calculate the Intersection over Union (IoU) between two binary masks."""
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()
    iou = intersection / (union + 1e-7)
    return iou

class VQVAEInference:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.vqvae_model = None
        
    def load_vqvae_model(self, cfg):
        """Load VQ-VAE model"""
        try:
            # Update paths based on your actual structure
            base_path = "/home/tqlong/qtung/gen-model-boilerplate"
            vqvae_cfg_path = f"{base_path}/configs/model/vae/vq_vae_module.yaml"
            
            if os.path.exists(vqvae_cfg_path):
                vqvae_cfg = OmegaConf.load(vqvae_cfg_path)
            else:
                print(f"VQ-VAE config file not found: {vqvae_cfg_path}")
                return
            
            # Update paths to model weights
            encoder_path = f"{base_path}/src/ckpt_s256/vq_vae/encoder.pth"
            decoder_path = f"{base_path}/src/ckpt_s256/vq_vae/decoder.pth"
            vq_layer_path = f"{base_path}/src/ckpt_s256/vq_vae/vq_layer_1024.pth"
            
            # Instantiate model
            model = hydra.utils.instantiate(vqvae_cfg)
            vq_vae = model.net
            
            # Load pre-trained weights if they exist
            if os.path.exists(encoder_path):
                encoder_state_dict = torch.load(encoder_path, map_location=self.device)
                vq_vae.encoder.load_state_dict(encoder_state_dict)
                print("✓ VQ-VAE Encoder loaded")
            
            if os.path.exists(decoder_path):
                decoder_state_dict = torch.load(decoder_path, map_location=self.device)
                vq_vae.decoder.load_state_dict(decoder_state_dict)
                print("✓ VQ-VAE Decoder loaded")
                
            if os.path.exists(vq_layer_path):
                vq_layer_state_dict = torch.load(vq_layer_path, map_location=self.device)
                vq_vae.vq_layer.load_state_dict(vq_layer_state_dict)
                print("✓ VQ-VAE VQ Layer loaded")
            
            vq_vae.to(self.device)
            vq_vae.eval()
            self.vqvae_model = vq_vae
            print("✓ VQ-VAE model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load VQ-VAE model: {e}")
            self.vqvae_model = None
    
    def vqvae_inference(self, image):
        """Perform VQ-VAE inference"""
        if self.vqvae_model is None:
            print("VQ-VAE model not loaded")
            return None, None
            
        with torch.no_grad():
            try:
                # VQ-VAE forward pass
                reconstructed, _ = self.vqvae_model(image)
                
                # Create anomaly map
                anomaly_map = create_anomaly_heatmap(image, reconstructed)
                
                return reconstructed.cpu(), anomaly_map
                
            except Exception as e:
                print(f"VQ-VAE inference error: {e}")
                return None, None

def visualize_and_save_grid(original, latents, ldm_generated, ldm_mask, 
                           vqvae_generated, vqvae_mask, ground_truth_mask, 
                           number, save_dir):
    """
    Visualize and save 3-row grid: Input, Ours (LDM), VQ-VAE
    Each row has 5 columns: T1, T1ce, T2, FLAIR, Predicted Mask
    """
    original = original.cpu()
    ldm_generated = ldm_generated.cpu()
    vqvae_generated = vqvae_generated.cpu()
    
    # Create a figure with 3 rows and 5 columns
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    
    # Column headers
    col_headers = ['T1', 'T1ce', 'T2', 'FLAIR', 'Predicted Mask']
    for col, header in enumerate(col_headers):
        axes[0, col].text(0.5, 1.1, header, transform=axes[0, col].transAxes, 
                         ha='center', va='bottom', fontsize=32, fontweight='bold')
    
    # Row headers
    row_headers = ['Input', 'Ours', 'VQ-VAE']
    for row, header in enumerate(row_headers):
        axes[row, 0].text(-0.1, 0.5, header, transform=axes[row, 0].transAxes, 
                         ha='right', va='center', fontsize=32, fontweight='bold', rotation=90)
    
    # Row 1: Input images (original 4 channels + ground truth mask)
    for channel in range(4):
        img = normalize_image(original[0, channel])
        axes[0, channel].imshow(img, cmap='gray')
        axes[0, channel].axis('off')
    
    # Ground truth mask
    axes[0, 4].imshow(ground_truth_mask, cmap='gray')
    axes[0, 4].axis('off')
    
    # Row 2: LDM results (generated 4 channels + predicted mask)
    for channel in range(4):
        img = normalize_image(ldm_generated[0, channel])
        axes[1, channel].imshow(img, cmap='gray')
        axes[1, channel].axis('off')
    
    # LDM predicted mask
    axes[1, 4].imshow(ldm_mask[0], cmap='gray')
    axes[1, 4].axis('off')
    
    # Row 3: VQ-VAE results (generated 4 channels + predicted mask)
    for channel in range(4):
        img = normalize_image(vqvae_generated[0, channel])
        axes[2, channel].imshow(img, cmap='gray')
        axes[2, channel].axis('off')
    
    # VQ-VAE predicted mask
    axes[2, 4].imshow(vqvae_mask, cmap='gray')
    axes[2, 4].axis('off')
    
    # Calculate metrics
    ldm_dice = dice_score(ground_truth_mask, ldm_mask[0])
    ldm_iou = iou_score(ground_truth_mask, ldm_mask[0])
    vqvae_dice = dice_score(ground_truth_mask, vqvae_mask)
    vqvae_iou = iou_score(ground_truth_mask, vqvae_mask)
    
    # Adjust layout and save
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'comparison_grid_{number}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return ldm_dice, ldm_iou, vqvae_dice, vqvae_iou

@hydra.main(version_base="1.3", config_path="../configs", config_name="auroc_eval.yaml")
def main(cfg: DictConfig):
    logger.configure()
    
    logger.log("Setting up AUROC evaluation...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")
    
    # Instantiate datamodule 
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    test_loader = datamodule.val_dataloader()
    
    logger.log("Loading LDM model...")
    
    # Instantiate LDM model 
    ldm: LatentDiffusionModule = hydra.utils.instantiate(cfg.model)
    ldm.vae.to(device)
    ldm.model.to(device)
    ldm.eval()

    # Load classifier if specified in the config
    args = create_argparser().parse_args()
    classifier = None
    if hasattr(cfg, "classifier") and cfg.classifier.path:
        logger.log("Loading classifier...")
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        classifier.load_state_dict(torch.load(cfg.classifier.path, map_location=device))
        classifier.to(device)
        classifier.eval()
    
    # Initialize VQ-VAE inference
    logger.log("Loading VQ-VAE model...")
    vqvae_inference = VQVAEInference(device)
    vqvae_inference.load_vqvae_model(cfg)
    
    logger.log("Starting AUROC evaluation...")
    
    # For AUROC calculation - collect all ground truth and anomaly scores
    all_gt_flat = []  # All ground truth masks flattened
    all_ldm_scores_flat = []  # All LDM anomaly scores flattened
    all_vqvae_scores_flat = []  # All VQ-VAE anomaly scores flattened
    
    # For sample-wise metrics
    all_ldm_dice_scores = []
    all_ldm_iou_scores = []
    all_vqvae_dice_scores = []
    all_vqvae_iou_scores = []
    
    batch_size = cfg.data.batch_size
    num_samples = cfg.sampling.num_samples if hasattr(cfg.sampling, "num_samples") else -1
    save_dir = cfg.sampling.save_dir if hasattr(cfg.sampling, "save_dir") else "./samples"
    noise_level = cfg.sampling.noise_level if hasattr(cfg.sampling, "noise_level") else 500
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if cfg.classifier.use_classifier:
        classifier_scale = cfg.classifier.scale if hasattr(cfg, "classifier") and hasattr(cfg.classifier, "scale") else 100.0
    else:
        print("CLASSIFIER NOT USED")
        classifier_scale = 0.0
    
    for idx, batch in enumerate(tqdm(test_loader, desc="Sampling")):
        if idx >= cfg.evaluation.max_samples:
            break
        
        img, cond, mask, label = batch
        if mask is not None:
            if hasattr(cfg.sampling, "skip_normal") and cfg.sampling.skip_normal and label == 0:
                continue
            mask = mask.squeeze().cpu().numpy()  # ground truth mask
        else:
            print("No mask provided")
            mask = np.zeros((img.shape[2], img.shape[3]), dtype=bool)
        
        # Move data to device
        for i in range(len(batch)):
            if torch.is_tensor(batch[i]):
                batch[i] = batch[i].to(device)
        
        # Set class label for classifier guidance
        cond = {}
        if classifier is not None:
            classes = torch.randint(low=0, high=1, size=(batch_size,), device=device)
            cond["y"] = classes
        
        logger.log(f"Processing sample {idx}...")
        
        # LDM inference
        with torch.no_grad():
            best_dice = -1.0
            best_sample_results = None
            
            # Generate multiple LDM samples and pick the best
            for sample_idx in range(num_samples): 
                start_time = time.time()
                results = ldm.sample(
                    batch,
                    cond=cond,
                    classifier=classifier,
                    classifier_scale=classifier_scale,
                    noise_level=noise_level,
                    batch_size=batch_size,
                )
                print(f"Sample {sample_idx} generation time: {time.time() - start_time:.2f} seconds")
                
                current_dice = dice_score(mask, results["difftot_mask"][0])
                if current_dice > best_dice:
                    best_dice = current_dice
                    best_sample_results = results
            
            ldm_results = best_sample_results
        
        # VQ-VAE inference
        vqvae_reconstructed, vqvae_mask = vqvae_inference.vqvae_inference(batch[0])
        
        # Collect data for AUROC calculation
        # Flatten ground truth mask
        gt_flat = mask.flatten()
        all_gt_flat.append(gt_flat)
        
        # Flatten LDM anomaly scores (use the continuous difftot values)
        ldm_anomaly_scores = ldm_results["difftot"].flatten()
        all_ldm_scores_flat.append(ldm_anomaly_scores)
        
        if vqvae_reconstructed is not None and vqvae_mask is not None:
            # Flatten VQ-VAE anomaly scores
            # Create continuous anomaly map for VQ-VAE
            vqvae_anomaly_map = create_anomaly_heatmap(
                batch[0], 
                torch.tensor(vqvae_reconstructed).to(batch[0].device)
            )
            vqvae_scores_flat = vqvae_anomaly_map.flatten()
            all_vqvae_scores_flat.append(vqvae_scores_flat)
            
            # Create comparison grid every 100 samples
            if idx % 100 == 0:
                ldm_dice, ldm_iou, vqvae_dice, vqvae_iou = visualize_and_save_grid(
                    ldm_results["original_images"],
                    ldm_results["latents"],
                    ldm_results["generated_images"],
                    ldm_results["difftot_mask"],
                    vqvae_reconstructed,
                    vqvae_mask,
                    mask,
                    idx,
                    save_dir
                )
            else:
                # Calculate metrics without visualization
                ldm_dice = dice_score(mask, ldm_results["difftot_mask"][0])
                ldm_iou = iou_score(mask, ldm_results["difftot_mask"][0])
                vqvae_dice = dice_score(mask, vqvae_mask)
                vqvae_iou = iou_score(mask, vqvae_mask)
            
            all_ldm_dice_scores.append(ldm_dice)
            all_ldm_iou_scores.append(ldm_iou)
            all_vqvae_dice_scores.append(vqvae_dice)
            all_vqvae_iou_scores.append(vqvae_iou)
            
            logger.log(f"Sample {idx} - LDM Dice: {ldm_dice:.4f}, VQ-VAE Dice: {vqvae_dice:.4f}")
        else:
            logger.log(f"Sample {idx} - VQ-VAE inference failed, skipping")
    
    # Calculate and log results
    logger.log("Calculating final metrics...")
    
    # Calculate global AUROC (toàn dataset)
    if all_gt_flat and all_ldm_scores_flat:
        logger.log("Calculating global AUROC...")
        
        # Concatenate all flattened arrays
        global_y_true = np.concatenate(all_gt_flat)
        global_ldm_scores = np.concatenate(all_ldm_scores_flat)
        
        # Calculate LDM AUROC
        ldm_auroc = -1
        try:
            ldm_auroc = roc_auc_score(global_y_true, global_ldm_scores)
            logger.log(f"LDM Global AUROC: {ldm_auroc:.4f}")
        except Exception as e:
            logger.log(f"LDM AUROC calculation failed: {e}")
        
        # Calculate VQ-VAE AUROC if available
        vqvae_auroc = None
        if all_vqvae_scores_flat:
            try:
                global_vqvae_scores = np.concatenate(all_vqvae_scores_flat)
                vqvae_auroc = roc_auc_score(global_y_true, global_vqvae_scores)
                logger.log(f"VQ-VAE Global AUROC: {vqvae_auroc:.4f}")
            except Exception as e:
                vqvae_auroc = None
                logger.log(f"VQ-VAE AUROC calculation failed: {e}")
    
    # Calculate mean sample-wise metrics
    mean_ldm_dice = np.mean(all_ldm_dice_scores) if all_ldm_dice_scores else 0.0
    mean_ldm_iou = np.mean(all_ldm_iou_scores) if all_ldm_iou_scores else 0.0
    mean_vqvae_dice = np.mean(all_vqvae_dice_scores) if all_vqvae_dice_scores else 0.0
    mean_vqvae_iou = np.mean(all_vqvae_iou_scores) if all_vqvae_iou_scores else 0.0
    
    # Log final results
    logger.log("=== FINAL RESULTS ===")
    logger.log(f"Processed {len(all_ldm_dice_scores)} samples")
    logger.log(f"LDM - Mean Dice: {mean_ldm_dice:.4f}, Mean IoU: {mean_ldm_iou:.4f}")
    if ldm_auroc != -1:
        logger.log(f"LDM Global AUROC: {ldm_auroc:.4f}")
    
    if all_vqvae_dice_scores:
        logger.log(f"VQ-VAE - Mean Dice: {mean_vqvae_dice:.4f}, Mean IoU: {mean_vqvae_iou:.4f}")
        if vqvae_auroc is not None:
            logger.log(f"VQ-VAE Global AUROC: {vqvae_auroc:.4f}")
    
    # Save results to file
    results_file = os.path.join(save_dir, "auroc_results.txt")
    with open(results_file, "w") as f:
        f.write("AUROC Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Processed samples: {len(all_ldm_dice_scores)}\n")
        f.write(f"Noise level (L): {noise_level}\n")
        f.write(f"Classifier scale: {classifier_scale}\n")
        f.write(f"Skip normal samples: {hasattr(cfg.sampling, 'skip_normal') and cfg.sampling.skip_normal}\n")
        f.write("\n")
        
        f.write("LDM Results:\n")
        f.write(f"  Mean Dice: {mean_ldm_dice:.4f}\n")
        f.write(f"  Mean IoU: {mean_ldm_iou:.4f}\n")
        if ldm_auroc != -1:
            f.write(f"  Global AUROC: {ldm_auroc:.4f}\n")
        f.write("\n")
        
        if all_vqvae_dice_scores:
            f.write("VQ-VAE Results:\n")
            f.write(f"  Mean Dice: {mean_vqvae_dice:.4f}\n")
            f.write(f"  Mean IoU: {mean_vqvae_iou:.4f}\n")
            if vqvae_auroc is not None:
                f.write(f"  Global AUROC: {vqvae_auroc:.4f}\n")
        
        f.write("\nIndividual Sample Results:\n")
        for i in range(len(all_ldm_dice_scores)):
            f.write(f"Sample {i+1}: LDM Dice={all_ldm_dice_scores[i]:.4f}, IoU={all_ldm_iou_scores[i]:.4f}")
            if i < len(all_vqvae_dice_scores):
                f.write(f" | VQ-VAE Dice={all_vqvae_dice_scores[i]:.4f}, IoU={all_vqvae_iou_scores[i]:.4f}")
            f.write("\n")
    
    logger.log(f"Results saved to: {results_file}")
    logger.log("AUROC evaluation completed!")

if __name__ == "__main__":
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
    main()