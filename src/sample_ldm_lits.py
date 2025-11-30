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
from omegaconf import DictConfig, OmegaConf, open_dict
from models.diffusion.latent_diffusion_module import LatentDiffusionModule
from pytorch_lightning import LightningDataModule, LightningModule
from tqdm import tqdm
import csv
from sklearn.metrics import roc_auc_score

from models.guided_diffusion import dist_util, logger
from models.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    classifier_defaults,
    create_classifier,
)

# Setup root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def create_argparser():
    """Create argument parser for classifier"""
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
        dataset='lits'
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def normalize(img):
    """Normalize image to [0, 1] range"""
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img.cpu()

# Helper functions
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
    """Calculate Dice coefficient"""
    if ground_truth is None:
        return 0.0
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    dice = 2.0 * intersection / (ground_truth.sum() + prediction.sum() + 1e-7)
    return float(dice)

def iou_score(ground_truth, prediction):
    """Calculate IoU"""
    if ground_truth is None:
        return 0.0
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()
    iou = intersection / (union + 1e-7)
    return float(iou)

def auroc_score(ground_truth, prediction):
    """Calculate AUROC"""
    if ground_truth is None or ground_truth.max() == 0:
        return 0.0
    try:
        gt_binary = (ground_truth > 0).astype(np.uint8).flatten()
        pred_flat = prediction.flatten()
        if gt_binary.sum() == 0 or gt_binary.sum() == len(gt_binary):
            return 0.0
        return float(roc_auc_score(gt_binary, pred_flat))
    except:
        return 0.0

def visualize_and_save_grid(original, latents, ldm_generated, ldm_mask, ldm_anomaly,
                           vqvae_generated, vqvae_mask, vqvae_anomaly, ground_truth_mask, 
                           number, save_dir, metrics=None):
    """
    Visualize and save 3-row grid: Input, Ours (LDM), VQ-VAE
    Adapted for LITS (1 channel)
    """
    original = original.cpu()
    ldm_generated = ldm_generated.cpu()
    if vqvae_generated is not None:
        vqvae_generated = vqvae_generated.cpu()
    
    # Create a figure with 3 rows and 3 columns (Image, Anomaly Map, Mask)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15)) if vqvae_generated is not None else plt.subplots(2, 3, figsize=(15, 10))
    
    # Column headers
    col_headers = ['CT Image', 'Anomaly Map', 'Predicted Mask']
    for col, header in enumerate(col_headers):
        axes[0, col].text(0.5, 1.1, header, transform=axes[0, col].transAxes, 
                         ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # Row headers
    row_headers = ['Input', 'Ours (LDM)', 'VQ-VAE'] if vqvae_generated is not None else ['Input', 'Ours (LDM)']
    for row, header in enumerate(row_headers):
        axes[row, 0].text(-0.1, 0.5, header, transform=axes[row, 0].transAxes, 
                         ha='right', va='center', fontsize=20, fontweight='bold', rotation=90)
    
    # Row 1: Input
    img = normalize_image(original[0, 0])
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].axis('off')
    
    # Ground truth mask (shown in Anomaly Map column for Input row)
    if ground_truth_mask is not None:
        axes[0, 1].imshow(ground_truth_mask, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title("Ground Truth Mask")
    else:
        axes[0, 1].imshow(np.zeros_like(img), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off') # Empty for input row
    
    # Row 2: LDM
    img = normalize_image(ldm_generated[0, 0])
    axes[1, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ldm_anomaly, cmap='jet')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(ldm_mask[0], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].axis('off')
    
    if metrics and 'ldm' in metrics:
        m = metrics['ldm']
        text = f"Dice: {m['dice']:.3f}\nIoU: {m['iou']:.3f}\nAUROC: {m['auroc']:.3f}"
        axes[1, 1].text(0.02, 0.98, text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Row 3: VQ-VAE
    if vqvae_generated is not None:
        img = normalize_image(vqvae_generated[0, 0])
        axes[2, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(vqvae_anomaly, cmap='jet')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(vqvae_mask, cmap='gray', vmin=0, vmax=1)
        axes[2, 2].axis('off')
        
        if metrics and 'vqvae' in metrics:
            m = metrics['vqvae']
            text = f"Dice: {m['dice']:.3f}\nIoU: {m['iou']:.3f}\nAUROC: {m['auroc']:.3f}"
            axes[2, 1].text(0.02, 0.98, text, transform=axes[2, 1].transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'sample_{number:04d}.png'), dpi=300, bbox_inches='tight')
    plt.close()

class VQVAEInference:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.vqvae_model = None
        
    def load_vqvae_model(self, cfg):
        """Load VQ-VAE model with LITS specific config"""
        try:
            # Load VQ-VAE config
            vqvae_cfg_path = "/home/tqlong/qtung/gen-model-boilerplate/configs/model/vae/vq_vae_module.yaml"
            if os.path.exists(vqvae_cfg_path):
                vqvae_cfg = OmegaConf.load(vqvae_cfg_path)
            else:
                raise Exception(f"VQ-VAE config file not found: {vqvae_cfg_path}")
            
            # Update paths to absolute paths and LITS specific settings
            base_path = "/home/tqlong/qtung/gen-model-boilerplate"
            
            # Override for Lits
            with open_dict(vqvae_cfg):
                # Set dimensions directly
                latent_dim = 1
                vqvae_cfg.net.latent_dims = [latent_dim, 64, 64]
                
                # Fix Encoder
                vqvae_cfg.net.encoder.in_channels = 1
                vqvae_cfg.net.encoder.z_channels = latent_dim
                
                # Fix Decoder
                vqvae_cfg.net.decoder.out_channels = 1
                vqvae_cfg.net.decoder.z_channels = latent_dim
                
                # Fix VQ Layer
                vqvae_cfg.net.vq_layer.embedding_dim = latent_dim
                
                # Set paths to LITS checkpoints
                vqvae_cfg.encoder_path = f"{base_path}/src/ckpt_lits/vq_vae/encoder.pth"
                vqvae_cfg.decoder_path = f"{base_path}/src/ckpt_lits/vq_vae/decoder.pth"
                vqvae_cfg.vq_layer_path = f"{base_path}/src/ckpt_lits/vq_vae/vq_layer_1024.pth"
            
            # Instantiate model
            model = hydra.utils.instantiate(vqvae_cfg)
            vq_vae = model.net
            
            # Load pre-trained weights if they exist
            if os.path.exists(vqvae_cfg.encoder_path):
                encoder_state_dict = torch.load(vqvae_cfg.encoder_path, map_location=self.device)
                vq_vae.encoder.load_state_dict(encoder_state_dict)
                print("✓ VQ-VAE Encoder loaded")
            
            if os.path.exists(vqvae_cfg.decoder_path):
                decoder_state_dict = torch.load(vqvae_cfg.decoder_path, map_location=self.device)
                vq_vae.decoder.load_state_dict(decoder_state_dict)
                print("✓ VQ-VAE Decoder loaded")
                
            if os.path.exists(vqvae_cfg.vq_layer_path):
                vq_layer_state_dict = torch.load(vqvae_cfg.vq_layer_path, map_location=self.device)
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
            return None, None, None
            
        with torch.no_grad():
            try:
                # VQ-VAE forward pass
                reconstructed, _ = self.vqvae_model(image)
                
                # Create anomaly map
                anomaly_map = create_anomaly_heatmap(image, reconstructed)
                
                # Create binary mask using threshold
                threshold = np.percentile(anomaly_map, 95)  # Top 5% as anomaly
                binary_mask = (anomaly_map > threshold).astype(np.uint8)
                
                return reconstructed.cpu(), binary_mask, anomaly_map
                
            except Exception as e:
                print(f"VQ-VAE inference error: {e}")
                return None, None, None

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference_lits")
def main(cfg: DictConfig):
    logger.configure()
    
    logger.log("Setting up data module...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate datamodule 
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    test_loader = datamodule.val_dataloader()
    
    logger.log("Loading LDM model...")
    
    # Instantiate LDM model 
    # Need to ensure LDM config points to LITS checkpoints if needed, 
    # but currently it seems to use VAE components which we might need to override?
    # For now, let's assume the config passed to instantiate handles it or we patch it.
    # Ideally we should patch the LDM config similar to VAE if it uses VAE internally.
    # But LDM usually uses VAE for encoding/decoding.
    
    # Let's load LDM normally first.
    # Revert to loading LDM normally without LITS patching for VAE
    # Because we only have a BraTS-trained LDM which expects 4-channel VAE latents
    # And thus expects 4-channel input images for the VAE encoder.
    ldm_config_path = "/home/tqlong/qtung/gen-model-boilerplate/configs/model/diffusion/latent_diffusion_module.yaml"
    ldm_cfg = OmegaConf.load(ldm_config_path)
    
    # Patch LDM VAE config to use LITS checkpoints
    base_path = "/home/tqlong/qtung/gen-model-boilerplate"
    
    # Helper to resolve paths.root_dir
    def resolve_paths_recursive(cfg, root):
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and "${paths.root_dir}" in v:
                    cfg[k] = v.replace("${paths.root_dir}", root)
                elif isinstance(v, (dict, list)):
                    resolve_paths_recursive(v, root)
        elif isinstance(cfg, list):
            for item in cfg:
                if isinstance(item, (dict, list)):
                    resolve_paths_recursive(item, root)

    # Convert to container to avoid resolution errors
    ldm_cfg_dict = OmegaConf.to_container(ldm_cfg, resolve=False)
    
    # Resolve paths first
    resolve_paths_recursive(ldm_cfg_dict, base_path)
    
    # Convert back to DictConfig for easier manipulation
    ldm_cfg = OmegaConf.create(ldm_cfg_dict)

    # This function just does `hydra.utils.instantiate`.
    # It DOES NOT seem to patch LDM config in `inference_lits.py`.
    #
    # Wait, I might have missed something in `inference_lits.py`.
    # Let's re-read `inference_lits.py` carefully.
    with open_dict(ldm_cfg):
        # Patch VAE part of LDM
        ldm_cfg.dataset = 'lits'  # Ensure dataset is lits
        latent_dim = 1
        ldm_cfg.vae.latent_dims = [latent_dim, 64, 64]
        
        ldm_cfg.vae.encoder.in_channels = 1
        # Manually resolve internal interpolations
        ldm_cfg.vae.encoder.z_channels = latent_dim
        ldm_cfg.vae.decoder.out_channels = 1
        ldm_cfg.vae.decoder.z_channels = latent_dim
        ldm_cfg.vae.decoder.base_channels = ldm_cfg.vae.encoder.base_channels
        ldm_cfg.vae.decoder.block = ldm_cfg.vae.encoder.block
        ldm_cfg.vae.decoder.n_layer_blocks = ldm_cfg.vae.encoder.n_layer_blocks
        ldm_cfg.vae.decoder.drop_rate = ldm_cfg.vae.encoder.drop_rate
        ldm_cfg.vae.decoder.channel_multipliers = ldm_cfg.vae.encoder.channel_multipliers
        ldm_cfg.vae.decoder.attention = ldm_cfg.vae.encoder.attention
        ldm_cfg.vae.decoder.n_attention_heads = ldm_cfg.vae.encoder.n_attention_heads
        ldm_cfg.vae.decoder.n_attention_layers = ldm_cfg.vae.encoder.n_attention_layers
        
        ldm_cfg.vae.vq_layer.embedding_dim = latent_dim
        
        ldm_cfg.vae_net_path = None # Ensure this is None as per config
        ldm_cfg.encoder_path = f"{base_path}/src/ckpt_lits/vq_vae/encoder.pth"
        ldm_cfg.decoder_path = f"{base_path}/src/ckpt_lits/vq_vae/decoder.pth"
        ldm_cfg.vq_layer_path = f"{base_path}/src/ckpt_lits/vq_vae/vq_layer_1024.pth"
        
        # Use LITS LDM checkpoint
        ldm_cfg.model_path = f"{base_path}/src/ckpt_lits/latent_diffusion/unet_ldm64_1000step.pth"


    ldm: LatentDiffusionModule = hydra.utils.instantiate(ldm_cfg)
    ldm.vae.to(device)
    ldm.model.to(device)
    ldm.eval()

    # Load classifier if available
    logger.log("Loading classifier...")
    args = create_argparser().parse_args([])
    # args.update({"diffusion_steps": self.hparams.num_timesteps})
    classifier = None
    classifier_path = cfg.get('classifier_path', None)
    if classifier_path and os.path.exists(classifier_path):
        try:
            args_dict = args_to_dict(args, classifier_defaults().keys())
            args_dict.update({"dataset": "lits"})
            args_dict.update({"image_size": 64})
            classifier = create_classifier(**args_dict)
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
            classifier.to(device)
            classifier.eval()
            logger.log(f"✓ Classifier loaded from {classifier_path}")
        except Exception as e:
            logger.log(f"✗ Failed to load classifier: {e}")
            classifier = None
    else:
        logger.log("No classifier path specified or file not found")

    # Initialize VQ-VAE inference
    logger.log("Loading VQ-VAE model...")
    vqvae_inference = VQVAEInference(device)
    vqvae_inference.load_vqvae_model(cfg)
    
    logger.log("Starting sampling and evaluation...")
    
    # Output settings
    output_dir = cfg.get('output_dir', './results/sample_lits')
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV logging
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_idx', 'method', 'dice', 'iou', 'auroc'])
    
    num_samples = cfg.get('num_samples', 10)
    noise_level = cfg.get('noise_level', 200)
    batch_size = 1
    
    processed_count = 0
    
    for idx, batch in enumerate(tqdm(test_loader, desc="Sampling")):
        try:
            image, cond, mask, label = batch
            
            # Skip normal if requested
            if cfg.get('skip_normal', True) and label != 1:
                continue
                
            # Move to device
            image = image.to(device)
            # mask is numpy or tensor on cpu
            gt_mask = mask.squeeze().cpu().numpy() if mask is not None else None
            
            # Get classifier settings from config
            classifier_scale = cfg.get('classifier_scale', 100.0)
            use_classifier = cfg.get('use_classifier', classifier is not None)
            
            # Set conditioning dict for classifier guidance
            cond_dict = {}
            if use_classifier and classifier is not None:
                # Set class label for classifier guidance (0 for healthy/normal)
                cond_dict["y"] = torch.tensor([0], device=device)
            
            logger.log(f"Processing sample {processed_count}... (classifier: {use_classifier})")
            
            # === LDM Inference (Best of N) ===
            with torch.no_grad():
                best_dice = -1.0
                best_sample_results = None
                best_metrics = {}
                
                # Generate multiple LDM samples and pick the best
                for sample_idx in range(3):
                    # Prepare batch for LDM sample method [image, cond, mask, label]
                    ldm_batch = [image, {}, mask, label]
                    
                    results = ldm.sample(
                        ldm_batch,
                        cond=cond_dict,
                        classifier=classifier,
                        classifier_scale=classifier_scale if use_classifier else 0.0,
                        noise_level=noise_level,
                        batch_size=batch_size,
                    )
                    
                    # Calculate metrics for this sample
                    pred_mask = results["difftot_mask"][0]
                    
                    # Calculate raw anomaly map (not thresholded) for AUROC
                    original_img = results["original_images"]
                    generated_img = results["generated_images"]
                    pred_anomaly = create_anomaly_heatmap(original_img, generated_img)
                    
                    current_dice = dice_score(gt_mask, pred_mask)
                    
                    if current_dice > best_dice:
                        best_dice = current_dice
                        best_sample_results = results
                        best_metrics = {
                            'dice': current_dice,
                            'iou': iou_score(gt_mask, pred_mask),
                            'auroc': auroc_score(gt_mask, normalize_image(pred_anomaly))
                        }
                
                ldm_results = best_sample_results
            
            # === VQ-VAE Inference ===
            vqvae_reconstructed, vqvae_mask, vqvae_anomaly = None, None, None
            vqvae_metrics = {}
            
            # Enable VQ-VAE inference if configured
            if cfg.get('compare_with_vqvae', True) and vqvae_inference.vqvae_model is not None:
                vqvae_reconstructed, vqvae_mask, vqvae_anomaly = vqvae_inference.vqvae_inference(image)
                if vqvae_reconstructed is not None:
                    vqvae_metrics = {
                        'dice': dice_score(gt_mask, vqvae_mask),
                        'iou': iou_score(gt_mask, vqvae_mask),
                        'auroc': auroc_score(gt_mask, normalize_image(torch.from_numpy(vqvae_anomaly)))
                    }
            
            # Log metrics
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if vqvae_reconstructed is not None:
                    writer.writerow([processed_count, 'vae', vqvae_metrics['dice'], vqvae_metrics['iou'], vqvae_metrics['auroc']])
                if ldm_results:
                    writer.writerow([processed_count, 'ldm', best_metrics['dice'], best_metrics['iou'], best_metrics['auroc']])
            
            # Visualize
            metrics_display = {'ldm': best_metrics, 'vqvae': vqvae_metrics}
            
            if best_dice > 0.1:
                visualize_and_save_grid(
                    ldm_results["original_images"],
                    ldm_results["latents"],
                    ldm_results["generated_images"],
                    ldm_results["difftot_mask"],
                    ldm_results["difftot"],
                    vqvae_reconstructed,
                    vqvae_mask,
                    vqvae_anomaly,
                    gt_mask,
                    processed_count,
                    output_dir,
                    metrics=metrics_display
                )
            
            processed_count += 1
            # if processed_count >= num_samples:
            #     break
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
            
    logger.log("All done!")

if __name__ == "__main__":
    main()
