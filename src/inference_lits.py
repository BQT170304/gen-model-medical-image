import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pyrootutils
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.metrics import roc_auc_score
import pandas as pd
import csv

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def calculate_dice_score(pred_mask, gt_mask, threshold=0.5):
    """Calculate Dice coefficient
    
    Args:
        pred_mask: Predicted anomaly map (continuous values)
        gt_mask: Ground truth binary mask
        threshold: Threshold to binarize prediction
    
    Returns:
        Dice score (float)
    """
    if gt_mask is None or gt_mask.max() == 0:
        return 0.0
    
    # Binarize prediction
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Calculate Dice
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    dice = (2.0 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-7)
    
    return float(dice)

def calculate_iou_score(pred_mask, gt_mask, threshold=0.5):
    """Calculate Intersection over Union (IoU)
    
    Args:
        pred_mask: Predicted anomaly map (continuous values)
        gt_mask: Ground truth binary mask
        threshold: Threshold to binarize prediction
    
    Returns:
        IoU score (float)
    """
    if gt_mask is None or gt_mask.max() == 0:
        return 0.0
    
    # Binarize prediction
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Calculate IoU
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / (union + 1e-7)
    
    return float(iou)

def calculate_auroc_score(pred_mask, gt_mask):
    """Calculate Area Under ROC Curve
    
    Args:
        pred_mask: Predicted anomaly map (continuous values)
        gt_mask: Ground truth binary mask
    
    Returns:
        AUROC score (float)
    """
    if gt_mask is None or gt_mask.max() == 0:
        return 0.0
    
    try:
        gt_binary = (gt_mask > 0).astype(np.uint8).flatten()
        pred_flat = pred_mask.flatten()
        
        # Need at least one positive sample
        if gt_binary.sum() == 0 or gt_binary.sum() == len(gt_binary):
            return 0.0
        
        auroc = roc_auc_score(gt_binary, pred_flat)
        return float(auroc)
    except Exception as e:
        print(f"AUROC calculation error: {e}")
        return 0.0

def normalize_image(img):
    """Normalize image to [0, 1] range"""
    _min = img.min()
    _max = img.max()
    if _max > _min:
        return (img - _min) / (_max - _min)
    return img

def create_anomaly_heatmap(original, reconstructed, method='mse'):
    """Create anomaly heatmap from original and reconstructed images"""
    if method == 'mse':
        diff = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
    elif method == 'l1':
        diff = torch.mean(torch.abs(original - reconstructed), dim=1, keepdim=True)
    else:
        diff = torch.mean(torch.abs(original - reconstructed), dim=1, keepdim=True)
    
    return diff.squeeze().cpu().numpy()

def plot_comparison_grid(data_dict, save_path, sample_idx, metrics=None):
    """Plot comparison grid similar to the provided image format
    
    Args:
        data_dict: Dictionary with inference results
        save_path: Path to save the plot
        sample_idx: Sample index
        metrics: Dictionary with metrics for each method
    """
    
    # Lits has 1 channel (CT)
    # Columns: CT, Anomaly Map
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    
    # Column headers
    col_headers = ['CT', 'Anomaly Map']
    for col, header in enumerate(col_headers):
        axes[0, col].text(0.5, 1.1, header, transform=axes[0, col].transAxes, 
                         ha='center', va='bottom', fontsize=24, fontweight='bold')
    
    # Row headers
    row_headers = ['Input', 'Ours (CLDM)', 'VQVAE']
    for row, header in enumerate(row_headers):
        axes[row, 0].text(-0.1, 0.5, header, transform=axes[row, 0].transAxes, 
                         ha='right', va='center', fontsize=24, fontweight='bold', rotation=90)
    
    # Plot input images (first row)
    input_img = data_dict['input']  # Shape: [1, 1, H, W]
    
    # CT Image
    img = normalize_image(input_img[0, 0])
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].axis('off')
    
    # Input anomaly map (ground truth if available)
    if 'ground_truth_mask' in data_dict and data_dict['ground_truth_mask'] is not None:
        axes[0, 1].imshow(data_dict['ground_truth_mask'], cmap='gray', vmin=0, vmax=1)
    else:
        axes[0, 1].imshow(np.zeros_like(input_img[0, 0]), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].axis('off')
    
    # Plot results for each method
    methods = ['latent_diffusion', 'vae'] # Keys in data_dict
    
    for i, method_key in enumerate(methods):
        row = i + 1
        
        if method_key in data_dict and data_dict[method_key] is not None:
            # Plot reconstructed images
            reconstructed = data_dict[method_key]['reconstructed']
            img = normalize_image(reconstructed[0, 0])
            axes[row, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, 0].axis('off')
            
            # Plot anomaly map
            anomaly_map = data_dict[method_key]['anomaly_map']
            im = axes[row, 1].imshow(anomaly_map, cmap='jet', alpha=0.8)
            axes[row, 1].axis('off')
            
            # Add metrics text if available
            if metrics and method_key in metrics:
                metric_text = f"Dice: {metrics[method_key]['dice']:.3f}\n"
                metric_text += f"IoU: {metrics[method_key]['iou']:.3f}\n"
                metric_text += f"AUROC: {metrics[method_key]['auroc']:.3f}"
                axes[row, 1].text(0.02, 0.98, metric_text, 
                                transform=axes[row, 1].transAxes,
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # If method not available, show empty plots
            for col in range(2):
                axes[row, col].imshow(np.zeros_like(input_img[0, 0]), cmap='gray', vmin=0, vmax=1)
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class ModelInference:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        
    def load_vae_model(self, vae_config_path):
        """Load VQ-VAE model"""
        try:
            # Load config directly without interpolation
            vae_cfg = OmegaConf.load(vae_config_path)
            
            # Update paths to absolute paths for Lits
            base_path = "/home/tqlong/qtung/gen-model-boilerplate"
            
            # Override for Lits and fix interpolations
            with open_dict(vae_cfg):
                # Set dimensions directly to avoid interpolation errors
                latent_dim = 1
                vae_cfg.net.latent_dims = [latent_dim, 64, 64]
                
                # Fix Encoder
                vae_cfg.net.encoder.in_channels = 1
                vae_cfg.net.encoder.z_channels = latent_dim
                
                # Fix Decoder
                vae_cfg.net.decoder.out_channels = 1
                vae_cfg.net.decoder.z_channels = latent_dim
                
                # Fix VQ Layer
                vae_cfg.net.vq_layer.embedding_dim = latent_dim
                
                # Set paths
                vae_cfg.encoder_path = f"{base_path}/src/ckpt_lits/vq_vae/encoder.pth"
                vae_cfg.decoder_path = f"{base_path}/src/ckpt_lits/vq_vae/decoder.pth"
                vae_cfg.vq_layer_path = f"{base_path}/src/ckpt_lits/vq_vae/vq_layer_1024.pth"
            
            model = hydra.utils.instantiate(vae_cfg)
            vq_vae = model.net
            
            # Load pre-trained weights if they exist
            if os.path.exists(vae_cfg.encoder_path):
                encoder_state_dict = torch.load(vae_cfg.encoder_path, map_location=self.device)
                vq_vae.encoder.load_state_dict(encoder_state_dict)
            
            if os.path.exists(vae_cfg.decoder_path):
                decoder_state_dict = torch.load(vae_cfg.decoder_path, map_location=self.device)
                vq_vae.decoder.load_state_dict(decoder_state_dict)
                
            if os.path.exists(vae_cfg.vq_layer_path):
                vq_layer_state_dict = torch.load(vae_cfg.vq_layer_path, map_location=self.device)
                vq_vae.vq_layer.load_state_dict(vq_layer_state_dict)
            
            vq_vae.to(self.device)
            vq_vae.eval()
            self.models['vae'] = vq_vae
            print("✓ VAE model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load VAE model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_latent_diffusion_model(self, ldm_config_path):
        """Load Latent Diffusion model"""
        try:
            ldm_cfg = OmegaConf.load(ldm_config_path)
            
            base_path = "/home/tqlong/qtung/gen-model-boilerplate"
            
            # Override for Lits and fix interpolations
            with open_dict(ldm_cfg):
                ldm_cfg.dataset = "lits"
                ldm_cfg.num_timesteps = 1000
                
                # Fix VAE config inside LDM
                latent_dim = 1
                ldm_cfg.vae.latent_dims = [latent_dim, 64, 64]
                
                # Fix Encoder
                ldm_cfg.vae.encoder.in_channels = 1
                ldm_cfg.vae.encoder.z_channels = latent_dim
                
                # Fix Decoder
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
                
                # Fix VQ Layer
                ldm_cfg.vae.vq_layer.embedding_dim = latent_dim
                
                # Set paths
                ldm_cfg.encoder_path = f"{base_path}/src/ckpt_lits/vq_vae/encoder.pth"
                ldm_cfg.decoder_path = f"{base_path}/src/ckpt_lits/vq_vae/decoder.pth"
                ldm_cfg.vq_layer_path = f"{base_path}/src/ckpt_lits/vq_vae/vq_layer_1024.pth"
                ldm_cfg.model_path = f"{base_path}/src/ckpt_lits/latent_diffusion/unet_ldm64_1000step.pth"

            # Instantiate ldm_cfg directly (it is the model config)
            ldm = hydra.utils.instantiate(ldm_cfg)
            ldm.vae.to(self.device)
            ldm.model.to(self.device)
            ldm.eval()
            self.models['latent_diffusion'] = ldm
            print("✓ Latent Diffusion model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Latent Diffusion model: {e}")
            import traceback
            traceback.print_exc()
    
    def vae_inference(self, image):
        """Perform VAE inference"""
        if 'vae' not in self.models:
            return None
            
        with torch.no_grad():
            try:
                reconstructed, _ = self.models['vae'](image)
                anomaly_map = create_anomaly_heatmap(image, reconstructed)
                
                return {
                    'reconstructed': reconstructed.cpu(),
                    'anomaly_map': anomaly_map
                }
            except Exception as e:
                print(f"VAE inference error: {e}")
                return None
    
    def latent_diffusion_inference(self, batch, cond=None):
        """Perform Latent Diffusion inference"""
        if 'latent_diffusion' not in self.models:
            return None
            
        with torch.no_grad():
            try:
                results = self.models['latent_diffusion'].sample(
                    batch, 
                    cond=cond if cond else {},
                    noise_level=200,
                    batch_size=1
                )
                
                reconstructed = results['generated_images']
                anomaly_map = results['difftot']
                
                return {
                    'reconstructed': reconstructed.cpu(),
                    'anomaly_map': anomaly_map
                }
            except Exception as e:
                print(f"Latent diffusion inference error: {e}")
                return None

@hydra.main(
    version_base="1.3", 
    config_path="../configs", 
    config_name="inference_lits"
)
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize inference class
    inference = ModelInference(device)
    
    # Load models with explicit paths
    base_path = "/home/tqlong/qtung/gen-model-boilerplate"
    vae_config_path = f"{base_path}/configs/model/vae/vq_vae_module.yaml"
    ldm_config_path = f"{base_path}/configs/model/diffusion/latent_diffusion_module.yaml"
    
    if os.path.exists(vae_config_path):
        inference.load_vae_model(vae_config_path)
    
    if os.path.exists(ldm_config_path):
        inference.load_latent_diffusion_model(ldm_config_path)
        
    # Setup data
    try:
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()
        test_loader = datamodule.val_dataloader()
    except Exception as e:
        print(f"Error setting up datamodule: {e}")
        return
    
    # Output directory
    output_dir = cfg.get('output_dir', './results/inference_lits')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting inference comparison...")
    
    processed_count = 0
    for idx, batch in enumerate(tqdm(test_loader, desc="Processing samples")):
        try:
            image, cond, mask, label = batch
            
            # Skip samples without tumors if specified
            if cfg.get('skip_normal', True) and label != 1:
                continue
                
            # Move to device
            image = image.to(device)
            # mask is already a tensor on CPU, keep it that way for LDM
            
            # Move conditioning to device
            # if cond is not None and 'y' in cond:
            #     cond['y'] = cond['y'].to(device)
            
            # === TẠO HEALTHY CONDITION (THÊM MỚI) ===
            cond = {'y': torch.tensor([0], device=device)}  # 0 = healthy label
            
            # Collect results from all methods
            results_dict = {
                'input': image.cpu(),
                'ground_truth_mask': mask.squeeze().cpu().numpy() if mask is not None else None
            }
            
            # VAE inference
            vae_results = inference.vae_inference(image)
            if vae_results:
                results_dict['vae'] = vae_results
            
            # Latent Diffusion inference
            # Create batch with image on device for LDM
            ldm_batch = (image, cond, mask, label)
            ldm_results = inference.latent_diffusion_inference(ldm_batch, cond)
            if ldm_results:
                results_dict['latent_diffusion'] = ldm_results
            
            # Create comparison plot
            save_path = os.path.join(output_dir, f'comparison_sample_{processed_count:04d}.png')
            plot_comparison_grid(results_dict, save_path, processed_count)
            
            processed_count += 1
            print(f"Processed sample {processed_count}")
            
            # Check if we've processed enough samples
            # if processed_count >= cfg.get('num_samples', 10):
            #     break
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"Inference comparison completed! Results saved to {output_dir}")

if __name__ == "__main__":
    # Create config file
    config_yaml = """
defaults:
  - data: lits

data:
  batch_size: 1
  num_workers: 4
  full_dataset: true  # Load both healthy and unhealthy data

output_dir: /home/tqlong/qtung/gen-model-boilerplate/results/inference_lits
num_samples: 10
skip_normal: true  # Only process samples with tumors (skip label==0)
"""
    
    # Ensure config directory exists
    config_dir = "/home/tqlong/qtung/gen-model-boilerplate/configs"
    os.makedirs(config_dir, exist_ok=True)
    
    # Save config file
    config_path = os.path.join(config_dir, "inference_lits.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml)
    
    main()
