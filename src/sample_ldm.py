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

from models.guided_diffusion import dist_util, logger
from models.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    classifier_defaults,
    create_classifier,
)
from models.components.up_down import Decoder

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

def normalize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img.cpu()

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

def percentile_threshold(diff_map, percentile=95):
    threshold_value = np.percentile(diff_map, percentile)
    thresholded = np.where(diff_map > threshold_value, diff_map, 0)
    return thresholded

def dice_score(ground_truth, prediction):
    """
    Calculate the Dice coefficient between two binary masks.
    """
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    dice = 2.0 * intersection / (ground_truth.sum() + prediction.sum() + 1e-7)
    return dice

def iou_score(ground_truth, prediction):
    """
    Calculate the Intersection over Union (IoU) between two binary masks.
    """
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()
    iou = intersection / (union + 1e-7)
    return iou

def visualize_and_save(original, generated, predicted_mask, ground_truth_mask, number, save_dir, dice_threshold=0.75):
    """
    Save individual images for each sample in separate folders.
    Only saves if dice score > dice_threshold.
    
    Args:
        original: Input image tensor [B, 4, H, W] (4 modalities)
        generated: Generated image tensor [B, 4, H, W]
        predicted_mask: Predicted mask [B, H, W]
        ground_truth_mask: Ground truth mask [H, W]
        number: Sample index
        save_dir: Base save directory
        dice_threshold: Minimum dice score to save
    
    Returns:
        dice_score: Computed dice score
    """
    # Calculate dice score first
    dice = dice_score(ground_truth_mask, predicted_mask[0])
    
    # Only save if dice > threshold
    if dice < dice_threshold:
        return dice
    
    # Create sample folder
    sample_folder = os.path.join(save_dir, f'sample_{number:04d}')
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    
    # Convert to CPU and numpy
    original = original.cpu()
    generated = generated.cpu()
    
    # Modality names
    modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    # Save input images (4 modalities)
    for i, modality in enumerate(modalities):
        img = normalize_image(original[0, i])
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, f'input_{modality}.png'), 
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # Save output images (4 modalities)
    for i, modality in enumerate(modalities):
        img = normalize_image(generated[0, i])
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, f'output_{modality}.png'), 
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # Save ground truth mask
    plt.figure(figsize=(8, 8))
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(sample_folder, 'gt_mask.png'), 
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save predicted mask
    plt.figure(figsize=(8, 8))
    plt.imshow(predicted_mask[0], cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(sample_folder, 'predicted_mask.png'), 
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save anomaly map
    anomaly_map = torch.mean(torch.abs(generated[0] - original[0]), dim=0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(anomaly_map, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(sample_folder, 'anomaly_map.png'), 
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save metrics file
    iou = iou_score(ground_truth_mask, predicted_mask[0])
    with open(os.path.join(sample_folder, 'metrics.txt'), 'w') as f:
        f.write(f'Sample {number}\n')
        f.write(f'Dice Score: {dice:.4f}\n')
        f.write(f'IoU Score: {iou:.4f}\n')
    
    print(f"Sample {number} saved - Dice: {dice:.4f}, IoU: {iou:.4f}")
    
    return dice

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
    
    # plt.suptitle(f"LDM - Dice: {ldm_dice:.4f}, IoU: {ldm_iou:.4f} | VQ-VAE - Dice: {vqvae_dice:.4f}, IoU: {vqvae_iou:.4f}", 
    #              fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'comparison_grid_{number}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return ldm_dice, ldm_iou, vqvae_dice, vqvae_iou

class VQVAEInference:
    def __init__(self, device='cuda:3'):
        self.device = device
        self.vqvae_model = None
        
    def load_vqvae_model(self, cfg):
        """Load VQ-VAE model"""
        try:
            # Load VQ-VAE config
            vqvae_cfg_path = "/home/tqlong/qtung/gen-model-boilerplate/configs/model/vae/vq_vae_module.yaml"
            if os.path.exists(vqvae_cfg_path):
                vqvae_cfg = OmegaConf.load(vqvae_cfg_path)
            else:
                raise Exception(f"VQ-VAE config file not found: {vqvae_cfg_path}")
            
            # Update paths to absolute paths
            base_path = "/home/tqlong/qtung/gen-model-boilerplate"
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
                
                # Create binary mask using threshold
                threshold = np.percentile(anomaly_map, 95)  # Top 5% as anomaly
                binary_mask = (anomaly_map > threshold).astype(np.uint8)
                
                return reconstructed.cpu(), binary_mask
                
            except Exception as e:
                print(f"VQ-VAE inference error: {e}")
                return None, None

@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_ldm.yaml")
def main(cfg: DictConfig):
    logger.configure()
    
    logger.log("Setting up data module...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    # logger.log("Loading VQ-VAE model...")
    # vqvae_inference = VQVAEInference(device)
    # vqvae_inference.load_vqvae_model(cfg)
    
    logger.log("Starting sampling and evaluation...")
    all_ldm_dice_scores = []
    all_ldm_iou_scores = []
    all_vqvae_dice_scores = []
    all_vqvae_iou_scores = []
    
    batch_size = cfg.data.batch_size
    num_samples = cfg.sampling.num_samples if hasattr(cfg.sampling, "num_samples") else -1
    save_dir = cfg.sampling.save_dir if hasattr(cfg.sampling, "save_dir") else "./samples"
    os.makedirs(save_dir, exist_ok=True)
    noise_level = cfg.sampling.noise_level if hasattr(cfg.sampling, "noise_level") else 500
    
    if cfg.classifier.use_classifier:
        classifier_scale = cfg.classifier.scale if hasattr(cfg, "classifier") and hasattr(cfg.classifier, "scale") else 100.0
    else:
        print("CLASSIFIER NOT USED")
        classifier_scale = 0.0
    
    # Define target index ranges
    target_ranges = [(100, 350), (800, 1500)]
    saved_samples = 0
    
    for idx, batch in enumerate(tqdm(test_loader, desc="Sampling")):
        # Check if current index is in target ranges
        in_target_range = idx == 300
        if not in_target_range:
            continue
            
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
            for sample_idx in range(3):  # Reduced from 8 for faster processing
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
        
        # Save individual images if dice > 0.75
        dice = visualize_and_save(
            ldm_results["original_images"],
            ldm_results["generated_images"],
            ldm_results["difftot_mask"],
            mask,
            idx,
            save_dir,
            dice_threshold=0.3
        )
        
        if dice > 0:
            saved_samples += 1
            all_ldm_dice_scores.append(dice)
            logger.log(f"Sample {idx} saved - Dice: {dice:.4f} (Total saved: {saved_samples})")
        else:
            logger.log(f"Sample {idx} skipped - Dice: {dice:.4f}")
    
    # Calculate and log mean scores
    if all_ldm_dice_scores:
        mean_ldm_dice = np.mean(all_ldm_dice_scores)
        mean_ldm_iou = np.mean(all_ldm_iou_scores)
        # mean_vqvae_dice = np.mean(all_vqvae_dice_scores)
        # mean_vqvae_iou = np.mean(all_vqvae_iou_scores)
        
        logger.log(f"Evaluation complete")
        logger.log(f"LDM - Mean Dice: {mean_ldm_dice:.4f}, Mean IoU: {mean_ldm_iou:.4f}")
        # logger.log(f"VQ-VAE - Mean Dice: {mean_vqvae_dice:.4f}, Mean IoU: {mean_vqvae_iou:.4f}")
        
        # Save results to a text file
        with open(os.path.join(save_dir, "comparison_results.txt"), "w") as f:
            f.write(f"LDM - Mean Dice: {mean_ldm_dice:.4f}, Mean IoU: {mean_ldm_iou:.4f}\n")
            # f.write(f"VQ-VAE - Mean Dice: {mean_vqvae_dice:.4f}, Mean IoU: {mean_vqvae_iou:.4f}\n")
            f.write("\nIndividual scores:\n")
            for i, (ldm_dice, ldm_iou) in enumerate(zip(all_ldm_dice_scores, all_ldm_iou_scores)):
                f.write(f"Sample {i}: LDM Dice={ldm_dice:.4f}, IoU={ldm_iou:.4f}\n")
    
    logger.log("All done!")

def compare_L():
    """
    Compare different L values (noise levels) on 10 images.
    For each image, run 7 times with L values: [10, 50, 100, 150, 200, 250, 500]
    Create visualization with 2 rows (input, output) and 7 columns (different L values)
    """
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    
    # Initialize with default config or load from file
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(config_name="sample_ldm.yaml")
    
    logger.configure()
    
    logger.log("Setting up data module for L comparison...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate datamodule 
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    test_loader = datamodule.val_dataloader()
    
    logger.log("Loading LDM model for L comparison...")
    
    # Instantiate LDM model 
    ldm = hydra.utils.instantiate(cfg.model)
    ldm.vae.to(device)
    ldm.model.to(device)
    ldm.eval()

    # Load classifier if specified in the config
    args = create_argparser().parse_args()
    classifier = None
    if hasattr(cfg, "classifier") and cfg.classifier.path:
        logger.log("Loading classifier...")
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        classifier.load_state_dict(torch.load(cfg.classifier.path))
        classifier.to(device)
        classifier.eval()
    
    L_values = [10, 50, 200, 500, 1000]
    batch_size = cfg.data.batch_size
    num_samples = cfg.sampling.num_samples if hasattr(cfg.sampling, "num_samples") else -1
    save_dir = cfg.sampling.save_dir if hasattr(cfg.sampling, "save_dir") else "./samples"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Classifier settings
    if hasattr(cfg, "classifier") and cfg.classifier.use_classifier:
        classifier_scale = cfg.classifier.scale if hasattr(cfg.classifier, "scale") else 100.0
    else:
        classifier_scale = 0.0
    
    logger.log("Starting L comparison...")
    
    # Process 10 images
    processed_images = 0
    for idx, batch in enumerate(tqdm(test_loader, desc="Processing images")):
        if processed_images >= 5:
            break
        if idx < 12:
            continue
            
        img, cond, mask, label = batch
        
        # Skip if no mask or if we want to skip normal samples
        if mask is not None:
            if hasattr(cfg.sampling, "skip_normal") and cfg.sampling.skip_normal and label == 0:
                continue
            mask = mask.squeeze().cpu().numpy()
        else:
            mask = np.zeros((img.shape[2], img.shape[3]), dtype=bool)
        
        # Move data to device
        for i in range(len(batch)):
            if torch.is_tensor(batch[i]):
                batch[i] = batch[i].to(device)
        
        # Set class label for classifier guidance
        cond_dict = {}
        if classifier is not None:
            classes = torch.randint(low=0, high=1, size=(batch_size,), device=device)
            cond_dict["y"] = torch.tensor([0]*batch_size, device=device) 
        
        logger.log(f"Processing image {processed_images + 1}/10...")
        
        # Store results for different L values
        results_dict = {}
        original_image = batch[0].cpu()  # Store original for visualization
        
        # Test each L value
        for L_val in L_values:
            logger.log(f"Testing L = {L_val}...")
            
            with torch.no_grad():
                results = ldm.sample(
                    batch,
                    cond=cond_dict,
                    classifier=classifier,
                    classifier_scale=classifier_scale,
                    noise_level=L_val,  # This is the L parameter
                    batch_size=batch_size,
                )
                
                # Store results
                results_dict[L_val] = {
                    'generated': results["generated_images"].cpu(),
                    'mask': results["difftot_mask"],
                    'anomaly_map': results["difftot"],
                    'dice': dice_score(mask, results["difftot_mask"][0])
                }
        
        # Create visualization
        visualize_L_comparison(
            original_image, 
            results_dict, 
            mask, 
            L_values, 
            processed_images + 1, 
            save_dir
        )
        
        processed_images += 1
    
    logger.log("L comparison complete!")

def visualize_L_comparison(original, results_dict, ground_truth_mask, L_values, image_num, save_dir):
    """
    Create visualization comparing different L values
    Top row: Input + Generated images for different L values
    Bottom row: Ground truth mask + Anomaly maps for different L values
    """
    # Create figure with 7 columns (Input + 6 L values)
    fig, axes = plt.subplots(2, 7, figsize=(30, 10))
    
    # Add spacing between rows
    plt.subplots_adjust(hspace=0.15, wspace=0.02)
    
    # Column headers
    headers = ['Input', 'L=10', 'L=50', 'L=100', 'L=200', 'L=500', 'L=1000']
    for col, header in enumerate(headers):
        axes[0, col].text(0.5, 1.02, header, transform=axes[0, col].transAxes, 
                         ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    # Row headers
    axes[0, 0].text(-0.08, 0.5, 'Images', transform=axes[0, 0].transAxes, 
                   ha='right', va='center', fontsize=28, fontweight='bold', rotation=90)
    axes[1, 0].text(-0.08, 0.5, 'Anomaly Map', transform=axes[1, 0].transAxes, 
                   ha='right', va='center', fontsize=28, fontweight='bold', rotation=90)
    
    # First column: Input image and ground truth mask
    img = normalize_image(original[0, 0])  # First channel
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[1, 0].imshow(ground_truth_mask, cmap='gray')
    axes[1, 0].axis('off')
    
    selected_L_values = [10, 50, 100, 200, 500, 1000]
    
    # Tạo một danh sách để lưu tất cả anomaly maps
    all_anomaly_maps = []
    
    # Trước tiên tính toán tất cả anomaly maps để chuẩn hóa cùng một thang đo
    for col, L_val in enumerate(selected_L_values, 1):
        if L_val not in results_dict:
            available_L = list(results_dict.keys())
            closest_L = min(available_L, key=lambda x: abs(x - L_val))
            L_val = closest_L
        
        generated = results_dict[L_val]['generated']
        
        # Tạo anomaly map từ original và generated
        original_tensor = original[0:1]  # Keep batch dimension
        generated_tensor = generated[0:1]
        anomaly_map = torch.mean(torch.abs(generated_tensor - original_tensor), dim=1).squeeze().cpu().numpy()
        all_anomaly_maps.append(anomaly_map)
    
    # Chuẩn hóa tất cả anomaly maps cùng một thang đo
    # Tìm giá trị min, max toàn cục
    global_min = min(map.min() for map in all_anomaly_maps)
    global_max = max(map.max() for map in all_anomaly_maps)
    
    # Remaining columns: Generated images and anomaly maps for different L values
    for col, L_val in enumerate(selected_L_values, 1):
        # If L_val not in results_dict, use closest available or skip
        if L_val not in results_dict:
            # Find closest L value
            available_L = list(results_dict.keys())
            closest_L = min(available_L, key=lambda x: abs(x - L_val))
            L_val = closest_L
        
        generated = results_dict[L_val]['generated']
        mask = results_dict[L_val]['mask']
        difftot = (original[0, :4, ...] - generated[0, ...]).mean(dim=0)
        
        # Top row: Generated image
        img = normalize_image(generated[0, 0])
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].axis('off')
        
        # Bottom row: Anomaly map (colored heatmap)
        anomaly_map = all_anomaly_maps[col-1]
        
        # Chuẩn hóa theo thang đo toàn cục
        normalized_map = (anomaly_map - global_min) / (global_max - global_min + 1e-10)
        
        # Display anomaly map with jet colormap (blue to red)
        # im = axes[1, col].imshow(normalized_map, cmap='plasma', vmin=0, vmax=1)
        im = axes[1, col].imshow(results_dict[L_val]['anomaly_map'], cmap='jet')
        axes[1, col].axis('off')
    
    # Remove title to match the figure style
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'L_comparison_image_{image_num}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save dice scores to text file
    scores_file = os.path.join(save_dir, f'dice_scores_image_{image_num}.txt')
    with open(scores_file, 'w') as f:
        f.write(f"Image {image_num} - Dice scores for different L values:\n")
        for L_val in selected_L_values:
            if L_val in results_dict:
                dice = results_dict[L_val]['dice']
                f.write(f"L = {L_val}: Dice = {dice:.4f}\n")
        
        # Find best L among available values
        available_dices = [(L, results_dict[L]['dice']) for L in selected_L_values if L in results_dict]
        if available_dices:
            best_L, best_dice = max(available_dices, key=lambda x: x[1])
            f.write(f"\nBest: L = {best_L} with Dice = {best_dice:.4f}\n")
            logger.log(f"Image {image_num} - Best L: {best_L} (Dice: {best_dice:.3f})")

if __name__ == "__main__":
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
    
    # Uncomment the line below to run L comparison instead of main
    # compare_L()
    main()
