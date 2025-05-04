import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pyrootutils
import hydra
from omegaconf import DictConfig, OmegaConf
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

def percentile_threshold(diff_map, percentile=80):
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

def visualize_and_save(original, latents, generated1, generated2, difftot_mask1, difftot_mask2,
                       ground_truth_mask, difftot, number, num_samples, save_dir):
    """
    Visualize and save comparison between original, generated images and masks.
    """
    original = original.cpu()
    generated1 = generated1.cpu()
    generated2 = generated2.cpu()
    
    # Create a figure and set of subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Original, Latents, Ground Truth, Difference Map
    axes[0, 0].imshow(original[0, 0, ...].numpy(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    
    axes[0, 1].imshow(latents[0, 0, ...].cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Latent Representation')
    
    axes[0, 2].imshow(ground_truth_mask, cmap='gray')
    axes[0, 2].set_title('Ground Truth Mask')
    
    # difftot = percentile_threshold((original[0, :4, ...] - generated1[0, ...]).mean(dim=0))
    # difftot = (original[0, :4, ...] - generated1[0, ...]).mean(dim=0)
    axes[0, 3].imshow(difftot, cmap='plasma')
    axes[0, 3].set_title('Difference Map')
    
    # Row 2: Generated1, Generated2, Difftot1, Difftot2
    axes[1, 0].imshow(generated1[0, 0, ...].numpy(), cmap='gray')
    axes[1, 0].set_title('Generated Image 1')
    
    axes[1, 1].imshow(generated2[0, 0, ...].numpy(), cmap='gray')
    axes[1, 1].set_title('Generated Image 2')
    
    axes[1, 2].imshow(difftot_mask1[0], cmap='gray')
    axes[1, 2].set_title('Thresholded Mask 1')
    
    axes[1, 3].imshow(difftot_mask2[0], cmap='gray')
    axes[1, 3].set_title('Thresholded Mask 2')
    
    # Calculate metrics
    dice = dice_score(ground_truth_mask, difftot_mask1)
    iou = iou_score(ground_truth_mask, difftot_mask1)
    dice2 = dice_score(ground_truth_mask, difftot_mask2)
    iou2 = iou_score(ground_truth_mask, difftot_mask2)
    and_dice = dice_score(ground_truth_mask, np.logical_and(difftot_mask1, difftot_mask2).astype(np.uint8))
    or_dice = dice_score(ground_truth_mask, np.logical_or(difftot_mask1, difftot_mask2).astype(np.uint8))
    dices = np.array([dice, dice2, and_dice, or_dice])
    max_idx = np.argmax(dices)
    if max_idx == 0:
        max_name = "Dice1"
        max_iou = iou
    elif max_idx == 1:
        max_name = "Dice2"
        max_iou = iou2
    elif max_idx == 2:
        max_name = "AND"
        max_iou = iou_score(ground_truth_mask, np.logical_and(difftot_mask1, difftot_mask2).astype(np.uint8))
    else:
        max_name = "OR"
        max_iou = iou_score(ground_truth_mask, np.logical_or(difftot_mask1, difftot_mask2).astype(np.uint8))
    print(f"Max Dice: {max_name} = {dices[max_idx]:.4f}")
    print(f"Max IoU: {max_name} = {max_iou:.4f}")
    
    plt.suptitle(f"Max Dice: {max_name} = {dices[max_idx]:.4f} | Max IoU: {max_iou:.4f}")
    
    
    # Adjust layout and save the figure
    plt.tight_layout()
    # if dices[max_idx] < 0.5 or dices[max_idx] > 0.85:
    # if dices[max_idx] > 0.5 or dices[max_idx] < 0.3:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'sample_{number}.png')) # _dice_{dice:.4f}_iou_{iou:.4f}
    plt.close()
    
    return dices[max_idx], max_iou

def analyze_brightness_stats(original, generated, number):
    """Compare statistical properties of original and generated images"""
    # Convert to numpy if needed
    if torch.is_tensor(original):
        original = original.cpu().numpy()
        generated = generated.cpu().numpy()
    
    # Calculate statistics
    stats = {
        "original_mean": np.mean(original),
        "generated_mean": np.mean(generated),
        "mean_ratio": np.mean(generated) / np.mean(original),
        "original_median": np.median(original),
        "generated_median": np.median(generated),
        "original_std": np.std(original),
        "generated_std": np.std(generated),
        "original_range": [np.min(original), np.max(original)],
        "generated_range": [np.min(generated), np.max(generated)]
    }
    
    print(f"Brightness stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    with open("brightness_stats.txt", "a") as f:
        f.write(f"Sample {number}:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value}\n")
    
    return stats

@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_ldm.yaml")
def main(cfg: DictConfig):
    logger.configure()
    
    logger.log("Setting up data module...")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # Instantiate datamodule 
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    test_loader = datamodule.val_dataloader()
    
    logger.log("Loading model...")
    
    # Instantiate model 
    ldm: LightningModule = hydra.utils.instantiate(cfg.model)
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
    
    logger.log("Starting sampling and evaluation...")
    all_dice_scores = []
    all_iou_scores = []
    batch_size = cfg.data.batch_size
    num_samples = cfg.sampling.num_samples if hasattr(cfg.sampling, "num_samples") else -1
    save_dir = cfg.sampling.save_dir if hasattr(cfg.sampling, "save_dir") else "./samples"
    noise_level = cfg.sampling.noise_level if hasattr(cfg.sampling, "noise_level") else 500
    classifier_scale = cfg.classifier.scale if hasattr(cfg, "classifier") and hasattr(cfg.classifier, "scale") else 100.0
    
    for idx, batch in enumerate(tqdm(test_loader, desc="Sampling")):
        if idx <= 100 or idx > 110:
            continue
        img, cond, mask, label = batch
        if mask is not None:
            if hasattr(cfg.sampling, "skip_normal") and cfg.sampling.skip_normal and label == 0:
                continue
            mask = mask.squeeze().cpu().numpy()  # ground truth mask
        else:
            # Handle case where mask isn't provided
            print("No mask provided")
            mask = np.zeros((img.shape[2], img.shape[3]), dtype=bool)
        
        # Move data to device
        for i in range(len(batch)):
            if torch.is_tensor(batch[i]):
                batch[i] = batch[i].to(device)
        
        # Set class label for classifier guidance
        if classifier is not None:
            # if hasattr(cfg.classifier, "use_label") and cfg.classifier.use_label:
            #     # Use the real label from the batch
            #     cond["y"] = batch[3].view(-1)
            # else:
            #     # Always use class 1 (anomaly) for sampling
            #     cond["y"] = torch.ones(batch_size, device=device, dtype=torch.long)
            classes = torch.randint(
                low=0, high=1, size=(batch_size,), device=device
            )
            cond["y"] = classes
        
        logger.log(f"Processing sample {idx}...")
        
        # Generate samples using the model
        with torch.no_grad():
            results = ldm.sample(
                batch,
                cond=cond,
                classifier=classifier,
                classifier_scale=classifier_scale,
                noise_level=noise_level,
                batch_size=batch_size,
            )
            
            results2 = ldm.sample(
                batch,
                cond=cond,
                classifier=classifier,
                classifier_scale=classifier_scale,
                noise_level=noise_level,
                batch_size=batch_size,
            )
        
        # Extract results
        latents = results["latents"]
        org_images = results["original_images"]
        difftot = results["difftot"]
        
        generated_images = results["generated_images"]
        difftot_mask = results["difftot_mask"]
        generated_images2 = results2["generated_images"]
        difftot_mask2 = results2["difftot_mask"]
        
        stats1 = analyze_brightness_stats(org_images, generated_images, idx)
        print('='*50)
        stats2 = analyze_brightness_stats(org_images, generated_images2, idx)
        
        # Visualize and evaluate
        dice, iou= visualize_and_save(
            org_images,
            latents,
            generated_images, generated_images2,
            difftot_mask, difftot_mask2,
            mask,
            difftot,
            idx,
            num_samples,
            save_dir
        )
        
        all_dice_scores.append(dice)
        all_iou_scores.append(iou)
        
        logger.log(f"Sample {idx} - Dice: {dice:.4f}, IoU: {iou:.4f}")
    
    # Calculate and log mean scores
    mean_dice = np.mean(all_dice_scores)
    mean_iou = np.mean(all_iou_scores)
    
    logger.log(f"Evaluation complete")
    logger.log(f"Mean Dice: {mean_dice:.4f}")
    logger.log(f"Mean IoU: {mean_iou:.4f}")
    
    # Save results to a text file
    with open(os.path.join(save_dir, "results.txt"), "w") as f:
        f.write(f"Mean Dice: {mean_dice:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write("\nIndividual scores:\n")
        for i, (dice, iou) in enumerate(zip(all_dice_scores, all_iou_scores)):
            f.write(f"Sample {i}: Dice={dice:.4f}, IoU={iou:.4f}\n")
    
    logger.log("All done!")

if __name__ == "__main__":
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
    main()
