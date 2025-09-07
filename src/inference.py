import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pyrootutils
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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

def plot_comparison_grid(data_dict, save_path, sample_idx):
    """Plot comparison grid similar to the provided image format"""
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 20))
    
    # Column headers
    col_headers = ['T1', 'T1ce', 'T2', 'FLAIR', 'Anomaly Map']
    for col, header in enumerate(col_headers):
        axes[0, col].text(0.5, 1.1, header, transform=axes[0, col].transAxes, 
                         ha='center', va='bottom', fontsize=24, fontweight='bold')
    
    # Row headers
    row_headers = ['Input', 'Ours (CLDM)', 'VQVAE']
    for row, header in enumerate(row_headers):
        axes[row, 0].text(-0.1, 0.5, header, transform=axes[row, 0].transAxes, 
                         ha='right', va='center', fontsize=24, fontweight='bold', rotation=90)
    
    # Plot input images (first row)
    input_img = data_dict['input']  # Shape: [1, 4, H, W]
    for channel in range(4):
        img = normalize_image(input_img[0, channel])
        axes[0, channel].imshow(img, cmap='gray')
        axes[0, channel].axis('off')
    
    # Input anomaly map (ground truth if available)
    if 'ground_truth_mask' in data_dict and data_dict['ground_truth_mask'] is not None:
        axes[0, 4].imshow(data_dict['ground_truth_mask'], cmap='gray')
    else:
        axes[0, 4].imshow(np.zeros_like(input_img[0, 0]), cmap='gray')
    axes[0, 4].axis('off')
    
    # Plot results for each method
    methods = ['Ours (CLDM)', 'VQVAE']
    
    for method_idx, method in enumerate(methods):
        row = method_idx + 1
        
        if method in data_dict and data_dict[method] is not None:
            # Plot reconstructed images for each channel
            reconstructed = data_dict[method]['reconstructed']
            for channel in range(4):
                img = normalize_image(reconstructed[0, channel])
                axes[row, channel].imshow(img, cmap='gray')
                axes[row, channel].axis('off')
            
            # Plot anomaly map
            anomaly_map = data_dict[method]['anomaly_map']
            im = axes[row, 4].imshow(anomaly_map, cmap='jet', alpha=0.8)
            axes[row, 4].axis('off')
        else:
            # If method not available, show empty plots
            for col in range(5):
                axes[row, col].imshow(np.zeros_like(input_img[0, 0]), cmap='gray')
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
            
            # Update paths to absolute paths
            base_path = "/data/hpc/qtung/gen-model-boilerplate"
            vae_cfg.encoder_path = f"{base_path}/src/ckpt_s256/vq_vae/encoder.pth"
            vae_cfg.decoder_path = f"{base_path}/src/ckpt_s256/vq_vae/decoder.pth"
            vae_cfg.vq_layer_path = f"{base_path}/src/ckpt_s256/vq_vae/vq_layer_1024.pth"
            
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
    
    def load_latent_diffusion_model(self, ldm_config_path):
        """Load Latent Diffusion model"""
        try:
            ldm_cfg = OmegaConf.load(ldm_config_path)
            ldm = hydra.utils.instantiate(ldm_cfg.model)
            ldm.vae.to(self.device)
            ldm.model.to(self.device)
            ldm.eval()
            self.models['latent_diffusion'] = ldm
            print("✓ Latent Diffusion model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Latent Diffusion model: {e}")
    
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
    config_name="inference_comparison"
)
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize inference class
    inference = ModelInference(device)
    
    # Load models with explicit paths
    vae_config_path = "/data/hpc/qtung/gen-model-boilerplate/configs/model/vae/vq_vae_module.yaml"
    ldm_config_path = "/data/hpc/qtung/gen-model-boilerplate/configs/model/diffusion/latent_diffusion_module.yaml"
    
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
    output_dir = cfg.get('output_dir', './inference_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting inference comparison...")
    
    for idx, batch in enumerate(tqdm(test_loader, desc="Processing samples")):
        if idx >= cfg.get('num_samples', 10):
            break
            
        try:
            image, cond, mask, label = batch
            
            # Skip normal cases if specified
            if cfg.get('skip_normal', True) and label == 0:
                continue
                
            # Move to device
            image = image.to(device)
            if mask is not None:
                mask = mask.squeeze().cpu().numpy()
            
            # Collect results from all methods
            results_dict = {
                'input': image.cpu(),
                'ground_truth_mask': mask if mask is not None else None
            }
            
            # VAE inference
            vae_results = inference.vae_inference(image)
            if vae_results:
                results_dict['vae'] = vae_results
            
            # Latent Diffusion inference
            ldm_results = inference.latent_diffusion_inference(batch, cond)
            if ldm_results:
                results_dict['latent_diffusion'] = ldm_results
            
            # Create comparison plot
            save_path = os.path.join(output_dir, f'comparison_sample_{idx:04d}.png')
            plot_comparison_grid(results_dict, save_path, idx)
            
            print(f"Processed sample {idx + 1}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"Inference comparison completed! Results saved to {output_dir}")

if __name__ == "__main__":
    # Create config file
    config_yaml = """
defaults:
  - data: brats2020

data:
  batch_size: 1
  num_workers: 4

output_dir: /data/hpc/qtung/gen-model-boilerplate/results/inference_comparison_2
num_samples: 10
skip_normal: true
"""
    
    # Ensure config directory exists
    config_dir = "/data/hpc/qtung/gen-model-boilerplate/configs"
    os.makedirs(config_dir, exist_ok=True)
    
    # Save config file
    config_path = os.path.join(config_dir, "inference_comparison.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml)
    
    main()