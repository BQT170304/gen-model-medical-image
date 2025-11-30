import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyrootutils
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Setup root
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import models
from src.models.vae.vae_module import VAEModule
from src.models.diffusion.latent_diffusion_module import LatentDiffusionModule
from src.models.classifier_module import ClassifierModule

def normalize_image(img):
    """Normalize image to [0, 1] range"""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
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

def percentile_threshold(diff_map, percentile=95):
    """Apply percentile thresholding to anomaly map"""
    threshold_value = np.percentile(diff_map, percentile)
    thresholded = np.where(diff_map > threshold_value, diff_map, 0)
    return (thresholded > 0).astype(np.uint8)

def dice_score(ground_truth, prediction):
    """Calculate the Dice coefficient between two binary masks"""
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    dice = 2.0 * intersection / (ground_truth.sum() + prediction.sum() + 1e-7)
    return dice

def iou_score(ground_truth, prediction):
    """Calculate the Intersection over Union (IoU) between two binary masks"""
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()
    iou = intersection / (union + 1e-7)
    return iou

class ModelInference:
    """Unified inference class for different model types"""
    
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.model = None
        self.model_type = None
        
    def load_model(self):
        """Load model based on configuration"""
        try:
            # Determine model type from config
            model_target = self.cfg.model._target_
            
            if "VAEModule" in model_target:
                self.model_type = "vae"
                self.model = hydra.utils.instantiate(self.cfg.model)
            elif "LatentDiffusionModule" in model_target:
                self.model_type = "ldm" 
                self.model = hydra.utils.instantiate(self.cfg.model)
            elif "ClassifierModule" in model_target:
                self.model_type = "classifier"
                self.model = hydra.utils.instantiate(self.cfg.model)
            else:
                raise ValueError(f"Unsupported model type: {model_target}")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ {self.model_type.upper()} model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model = None
    
    def infer(self, batch):
        """Perform inference based on model type"""
        if self.model is None:
            return None
            
        with torch.no_grad():
            try:
                if self.model_type == "vae":
                    return self._vae_infer(batch)
                elif self.model_type == "ldm":
                    return self._ldm_infer(batch)
                elif self.model_type == "classifier":
                    return self._classifier_infer(batch)
                else:
                    return None
                    
            except Exception as e:
                print(f"Inference error: {e}")
                return None
    
    def _vae_infer(self, batch):
        """VAE inference"""
        image, cond, mask, label = batch
        
        # VAE forward pass
        if hasattr(self.model, 'net'):
            reconstructed, _ = self.model.net(image)
        else:
            reconstructed, _ = self.model(image)
        
        # Create anomaly map
        anomaly_map = create_anomaly_heatmap(image, reconstructed)
        
        # Create predicted mask
        predicted_mask = percentile_threshold(anomaly_map, percentile=95)
        
        return {
            'original': image,
            'reconstructed': reconstructed,
            'anomaly_map': anomaly_map,
            'predicted_mask': predicted_mask,
            'ground_truth_mask': mask.squeeze().cpu().numpy() if mask is not None else None
        }
    
    def _ldm_infer(self, batch):
        """Latent Diffusion Model inference"""
        image, cond, mask, label = batch
        
        # Set up conditioning for LDM
        cond_dict = {}
        
        # LDM sampling
        results = self.model.sample(
            batch,
            cond=cond_dict,
            classifier=None,
            classifier_scale=0.0,
            noise_level=500,
            batch_size=1,
        )
        
        return {
            'original': results['original_images'],
            'reconstructed': results['generated_images'], 
            'anomaly_map': results['difftot'],
            'predicted_mask': results['difftot_mask'],
            'ground_truth_mask': mask.squeeze().cpu().numpy() if mask is not None else None
        }
    
    def _classifier_infer(self, batch):
        """Classifier inference"""
        # This would need to be implemented based on your classifier structure
        # For now, return None as placeholder
        return None

def save_brats_sample_individual(results, sample_idx, save_dir):
    """Save BraTS sample as individual images in a folder"""
    original = results['original'][0].cpu()  # [4, H, W]
    reconstructed = results['reconstructed'][0].cpu() if results['reconstructed'] is not None else None
    gt_mask = results['ground_truth_mask']
    pred_mask = results['predicted_mask']
    
    # Create sample folder
    sample_folder = os.path.join(save_dir, f'brats_sample_{sample_idx:03d}')
    os.makedirs(sample_folder, exist_ok=True)
    
    # Channel names
    channel_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    # Save input images
    for ch in range(4):
        img = normalize_image(original[ch])
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, f'input_{channel_names[ch].lower()}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save reconstructed images
    if reconstructed is not None:
        for ch in range(4):
            img = normalize_image(reconstructed[ch])
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(sample_folder, f'output_{channel_names[ch].lower()}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    # Save GT mask
    if gt_mask is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(gt_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, 'gt_mask.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save predicted mask
    if pred_mask is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, 'predicted_mask.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save anomaly map
    if 'anomaly_map' in results and results['anomaly_map'] is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(results['anomaly_map'], cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, 'anomaly_map.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate and save metrics
    if gt_mask is not None and pred_mask is not None:
        dice = dice_score(gt_mask, pred_mask)
        iou = iou_score(gt_mask, pred_mask)
        
        # Save metrics to text file
        with open(os.path.join(sample_folder, 'metrics.txt'), 'w') as f:
            f.write(f'Sample {sample_idx} Metrics:\n')
            f.write(f'Dice Score: {dice:.4f}\n')
            f.write(f'IoU Score: {iou:.4f}\n')
        
        return dice, iou
    
    return None, None

def save_lits_sample_individual(results, sample_idx, save_dir):
    """Save LiTS sample as individual images in a folder"""
    original = results['original'][0, 0].cpu()  # [H, W]
    reconstructed = results['reconstructed'][0, 0].cpu() if results['reconstructed'] is not None else None
    gt_mask = results['ground_truth_mask']
    pred_mask = results['predicted_mask']
    
    # Create sample folder
    sample_folder = os.path.join(save_dir, f'lits_sample_{sample_idx:03d}')
    os.makedirs(sample_folder, exist_ok=True)
    
    # Save input image
    plt.figure(figsize=(8, 8))
    plt.imshow(normalize_image(original), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(sample_folder, 'input.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save reconstructed image
    if reconstructed is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(normalize_image(reconstructed), cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, 'output.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save GT mask
    if gt_mask is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(gt_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, 'gt_mask.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save predicted mask
    if pred_mask is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, 'predicted_mask.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save difference map
    if reconstructed is not None:
        diff = np.abs(original.numpy() - reconstructed.numpy())
        plt.figure(figsize=(8, 8))
        plt.imshow(diff, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, 'difference_map.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save anomaly map
    if 'anomaly_map' in results and results['anomaly_map'] is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(results['anomaly_map'], cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, 'anomaly_map.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate and save metrics
    if gt_mask is not None and pred_mask is not None:
        dice = dice_score(gt_mask, pred_mask)
        iou = iou_score(gt_mask, pred_mask)
        
        # Save metrics to text file
        with open(os.path.join(sample_folder, 'metrics.txt'), 'w') as f:
            f.write(f'Sample {sample_idx} Metrics:\n')
            f.write(f'Dice Score: {dice:.4f}\n')
            f.write(f'IoU Score: {iou:.4f}\n')
        
        return dice, iou
    
    return None, None

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main visualization function"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate datamodule
    print("Setting up data module...")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    
    # Use test dataloader, fallback to val if not available
    try:
        dataloader = datamodule.test_dataloader()
        split_name = "test"
    except:
        dataloader = datamodule.val_dataloader()
        split_name = "val"
    
    print(f"Using {split_name} split with {len(dataloader)} batches")
    
    # Initialize model inference
    model_inference = ModelInference(cfg, device)
    model_inference.load_model()
    
    if model_inference.model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Determine model name for save directory
    model_name = model_inference.model_type
    if hasattr(cfg.model, 'net') and hasattr(cfg.model.net, '_target_'):
        model_name = cfg.model.net._target_.split('.')[-1].lower()
    
    # Create save directory
    save_dir = f"/home/tqlong/qtung/gen-model-boilerplate/results/visualize_{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}")
    
    # Determine dataset type
    dataset_name = cfg.data.dataset_name.lower() if hasattr(cfg.data, 'dataset_name') else 'unknown'
    
    # Process samples
    all_dice_scores = []
    all_iou_scores = []
    
    num_samples = getattr(cfg, 'num_samples', 10)  # Default to 10 samples
    
    print(f"Processing {num_samples} samples...")
    
    for idx, batch in enumerate(tqdm(dataloader, desc="Processing samples")):
        if idx >= num_samples:
            break
        
        # Move batch to device
        batch_device = []
        for item in batch:
            if torch.is_tensor(item):
                batch_device.append(item.to(device))
            else:
                batch_device.append(item)
        
        # Perform inference
        results = model_inference.infer(batch_device)
        
        if results is None:
            print(f"Inference failed for sample {idx}")
            continue
        
        # Save individual images based on dataset type
        dice, iou = None, None
        if 'brats' in dataset_name:
            dice, iou = save_brats_sample_individual(results, idx, save_dir)
        elif 'lits' in dataset_name:
            dice, iou = save_lits_sample_individual(results, idx, save_dir)
        else:
            # Default to LiTS format for unknown datasets
            dice, iou = save_lits_sample_individual(results, idx, save_dir)
        
        # Collect metrics
        if dice is not None and iou is not None:
            all_dice_scores.append(dice)
            all_iou_scores.append(iou)
    
    # Save summary metrics
    if all_dice_scores:
        mean_dice = np.mean(all_dice_scores)
        mean_iou = np.mean(all_iou_scores)
        std_dice = np.std(all_dice_scores)
        std_iou = np.std(all_iou_scores)
        
        summary_text = f"""Visualization Summary for {model_name}
Dataset: {dataset_name}
Number of samples: {len(all_dice_scores)}

Metrics:
Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}
Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}

Individual scores:
"""
        for i, (dice, iou) in enumerate(zip(all_dice_scores, all_iou_scores)):
            summary_text += f"Sample {i}: Dice={dice:.4f}, IoU={iou:.4f}\n"
        
        with open(os.path.join(save_dir, "summary.txt"), "w") as f:
            f.write(summary_text)
        
        print(f"\nSummary:")
        print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    
    print(f"\nVisualization complete! Results saved to: {save_dir}")

if __name__ == "__main__":
    main()