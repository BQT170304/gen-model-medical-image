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

def normalize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img.cpu()

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

def visualize_and_save(original, latents, generated, difftot_mask, ground_truth_mask, number, num_samples, save_dir):
    """
    Visualize and save comparison between original, generated images and masks.
    """
    # Ensure the images are in CPU and numpy format
    original = normalize(original)
    generated = normalize(generated)
    print(generated.min(), generated.max())
    
    # Create a figure and set of subplots
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    
    # Plot the images
    axes[0].imshow(generated[0, 0, ...].numpy(), cmap='gray')
    axes[0].set_title('Generated Image')
    
    axes[1].imshow(original[0, 0, ...].numpy(), cmap='gray')
    axes[1].set_title('Original Image')
    
    axes[2].imshow(latents[0, 0, ...].cpu().numpy(), cmap='gray')
    axes[2].set_title('Latent Representation')
    
    axes[3].imshow(ground_truth_mask, cmap='gray')
    axes[3].set_title('Ground Truth Mask')
    
    # Calculate the difference total (difftot) and plot it
    difftot=(original[0, :4,...] - generated[0, ...]).mean(dim=0)
    difftot = torch.clamp(difftot, min=0)
    difftot = difftot.numpy()
    axes[4].imshow(difftot, cmap='plasma')
    axes[4].set_title('Difference Map')
    
    axes[5].imshow(difftot_mask[0], cmap='gray')
    axes[5].set_title('Thresholded Mask')
    
    # Calculate metrics
    dice = dice_score(ground_truth_mask, difftot_mask)
    iou = iou_score(ground_truth_mask, difftot_mask)
    
    plt.suptitle(f'Dice: {dice:.4f}, IoU: {iou:.4f}')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    if dice < 0.5:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'sample_{number}_dice_{dice:.4f}_iou_{iou:.4f}.png'))
    plt.close()
    
    return dice, iou

@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_ldm.yaml")
def main(cfg: DictConfig):
    logger.configure()
    
    logger.log("Setting up data module...")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
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
        # Unpack the batch
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
        
        # Extract results
        latents = results["latents"]
        generated_images = results["generated_images"]
        difftot_mask = results["difftot"]
        
        # Visualize and evaluate
        dice, iou = visualize_and_save(
            img,
            latents,
            generated_images,
            difftot_mask,
            mask,
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
