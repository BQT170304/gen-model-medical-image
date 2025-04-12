from typing import Any, Dict, Tuple, Optional, List

import os
import wandb
import torch
import pytorch_lightning as pl
from torch import Tensor
from torchmetrics import MeanMetric
import numpy as np
import cv2
from skimage.morphology import remove_small_objects
from contextlib import contextmanager
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.up_down import Encoder, Decoder
from src.models.vae.net import BaseVAE, VQVAE
from src.utils.ema import LitEma
from src.models.guided_diffusion import dist_util
from src.models.guided_diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from src.models.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
wandb_key = '836d6530ee35ed83741702ab053789296fa1e4b5'
if wandb_key:
    wandb.login(key=wandb_key)
    
def dice_score(ground_truth, prediction):
    """
    Calculate the Dice coefficient between two binary masks.
    
    Parameters:
        ground_truth (np.ndarray): Ground truth binary mask.
        prediction (np.ndarray): Predicted binary mask.
    
    Returns:
        float: Dice coefficient.
    """
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    dice = 2.0 * intersection / (ground_truth.sum() + prediction.sum() + 1e-7)  # add small epsilon to avoid division by zero
    return dice

def iou_score(ground_truth, prediction):
    """
    Calculate the Intersection over Union (IoU) between two binary masks.
    
    Parameters:
        ground_truth (np.ndarray): Ground truth binary mask.
        prediction (np.ndarray): Predicted binary mask.
    
    Returns:
        float: IoU score.
    """
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()
    iou = intersection / (union + 1e-7)  # add small epsilon to avoid division by zero
    return iou

def morphology_close(mask, kernel_size=5, min_size=50):
    """
    Apply morphological closing and remove small objects from binary mask.
    
    Parameters:
        mask (numpy.ndarray): Binary mask (0-255, uint8)
        kernel_size (int): Size of the structuring element for closing
        min_size (int): Minimum size of objects to keep (pixels)
        
    Returns:
        numpy.ndarray: Cleaned binary mask (0-255, uint8)
    """
    assert mask.dtype == np.uint8, "Mask must be uint8"
    
    # Tạo kernel hình elip
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Áp dụng phép đóng (Morphological Closing)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask
    
class LatentDiffusionModule(pl.LightningModule):
    def __init__(
        self,
        vae: BaseVAE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        encoder_path: str,
        decoder_path: str,
        vq_layer_path: str,
        model_path: str = None,    
        use_ema: bool = False,
        num_timesteps: int = 1000,
    ) -> None:
        """Initialize the LatentDiffusionModule.

        Args:
            vae (BaseVAE): The VAE model used for encoding and decoding.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            encoder_path (str): Path to the encoder weights.
            decoder_path (str): Path to the decoder weights.
            vq_layer_path (str): Path to the VQ layer weights.
            use_ema (bool, optional): Whether to use EMA. Defaults to False.
            num_timesteps (int, optional): Number of diffusion timesteps. Defaults to 1000.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.dice = MeanMetric()
        self.iou = MeanMetric()
        
        # Load VAE components
        self.vae = vae
        
        # Load pre-trained weights if paths are provided
        if os.path.exists(encoder_path):
            self.vae.encoder.load_state_dict(torch.load(encoder_path))
            print("Loaded encoder weights")
        
        if os.path.exists(decoder_path):
            self.vae.decoder.load_state_dict(torch.load(decoder_path))
            print("Loaded decoder weights")
        
        if os.path.exists(vq_layer_path):
            self.vae.vq_layer.load_state_dict(torch.load(vq_layer_path))
            print("Loaded VQ layer weights")
            
        self.vae.eval()
        # Freeze VAE 
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # Create diffusion model
        defaults = self.get_default_args()
        args_dict = model_and_diffusion_defaults()
        args_dict.update(defaults)
        self.image_size = args_dict["image_size"]
        # print("args_dict: ", args_dict)
        self.model, self.diffusion = create_model_and_diffusion(
            **args_dict
        )
        if model_path is not None and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loaded diffusion model weights")
        
        # Create the schedule sampler
        self.schedule_sampler = create_named_schedule_sampler(
            "uniform", self.diffusion, maxt=num_timesteps
        )
        
        # Criterion for diffusion
        self.criterion = self.diffusion.training_losses
        
        # EMA
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            
        self.best_val_loss = float('inf')
        
        # self.classifier = torch.load("/data/hpc/qtung/gen-model-boilerplate/src/ckpt/classifier/classifier_ldm_32_4.pth")
        # self.classifier.eval()
    
    def get_default_args(self):
        """Return default arguments for the diffusion model."""
        defaults = dict(
            use_fp16=False,
            dataset='brats',
            image_size=64,
            num_channels=128,
            class_cond=True,
            dropout=0.0,
            learn_sigma=False,
            diffusion_steps=500,
            noise_schedule="linear",
            # timestep_respacing="ddim500",
        )
        
        return defaults
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
    
    def forward(self, imgs, cond):
        """Forward pass through the model.

        Args:
            imgs (Tensor): Input images.
            cond (Dict): Conditioning information.

        Returns:
            Tensor: Loss value.
        """
        t, weights = self.schedule_sampler.sample(imgs.shape[0], dist_util.dev())
        losses_dict = self.criterion(self.model, imgs, t, model_kwargs=cond)
        
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses_dict["loss"].detach()
            )
        
        losses = losses_dict[0] if isinstance(losses_dict, tuple) else losses_dict
        loss = (losses["loss"] * weights).mean()
        
        return loss
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
                    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.dice.reset()
        self.iou.reset()
    
    def training_step(self, batch: Tuple[Tensor, Dict], batch_idx: int) -> Tensor:
        """Training step.

        Args:
            batch (Tuple[Tensor, Dict]): Batch containing images and conditioning.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Loss value.
        """
        imgs, cond, _, labels = batch
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(imgs.float())
            # # Normalize latents to [-1, 1]
            # _max = latents.max()
            # _min = latents.min()
            # latents = (latents - _min) / (_max - _min)  # [0, 1]
            # latents = 2 * latents - 1  # [-1, 1]
        
        # Apply diffusion model on latents
        loss = self(latents, cond)
        # loss = self(latents, None) # Khong can condition
        
        # Update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Dict], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch (Tuple[Tensor, Dict]): Batch containing images and conditioning.
            batch_idx (int): Batch index.
        """
        imgs, cond, mask, labels = batch
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(imgs.float())       
            # # Normalize latents to [-1, 1]
            # _max = latents.max()
            # _min = latents.min()
            # latents = (latents - _min) / (_max - _min)  # [0, 1]
            # latents = 2 * latents - 1  # [-1, 1]
        
        loss = self(latents, cond)
        # loss = self(latents, None) # Khong can condition
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[Tensor, Dict], batch_idx: int) -> None:
        """Test step.

        Args:
            batch (Tuple[Tensor, Dict]): Batch containing images and conditioning.
            batch_idx (int): Batch index.
        """
        imgs, cond, mask, labels = batch
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(imgs.float())       
            # # Normalize latents to [-1, 1]
            # _max = latents.max()
            # _min = latents.min()
            # latents = (latents - _min) / (_max - _min)  # [0, 1]
            # latents = 2 * latents - 1  # [-1, 1]
        
        loss = self(latents, cond)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self.best_val_loss >= self.val_loss.compute():
            self.best_val_loss = self.val_loss.compute()
            print("SAVED!")
            if self.image_size == 64:
                save_path = "/data/hpc/qtung/gen-model-boilerplate/src/ckpt/latent_diffusion/unet_ldm64.pth"
            elif self.image_size == 32:
                save_path = "/data/hpc/qtung/gen-model-boilerplate/src/ckpt/latent_diffusion/unet_ldm32.pth"
            else:
                raise ValueError("Invalid image size")
            torch.save(
                self.model.state_dict(), 
                save_path
            )
        # if self.current_epoch % 10 == 0:
        #     wandb.log({
        #         "Dice": self.dice.compute(),
        #         "IoU": self.iou.compute(),  
        #     })
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers for training.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer and scheduler.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}
    
    def sample(self, batch, cond: Dict, classifier=None, classifier_scale=100, noise_level=500, batch_size: int = 1) -> Dict[str, Any]:
        """Generate samples using the diffusion model and calculate difftot as a binary mask.

        Args:
            batch: Input batch containing original images and other data.
            cond (Dict): Conditioning information.
            classifier: Optional classifier model for guided sampling.
            classifier_scale: Scale factor for classifier guidance.
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            Dict[str, Tensor]: Dictionary containing original images, generated images, and difftot mask.
        """
        original_images = batch[0]
        
        with torch.no_grad():
            latents = self.vae.encode(original_images.float())         
            # # Normalize latents to [-1, 1]
            # _max = latents.max()
            # _min = latents.min()
            # latents = (latents - _min) / (_max - _min)  # [0, 1]
            # latents = 2 * latents - 1  # [-1, 1]
        
        def model_fn(x, t, y=None):
            return self.model(x, t, y if cond.get("y") is not None else None)
        
        # Define conditional function for classifier guidance if classifier is provided
        cond_fn = None
        if classifier is not None:
            def cond_fn(x, t, y=None):
                assert y is not None
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = classifier(x_in, t)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), y.view(-1)]
                    grad = torch.autograd.grad(selected.sum(), x_in)[0]
                    return grad, grad * classifier_scale
        
        # Create a batch of original data in the format expected by ddim_sample_loop_known
        if classifier is not None:
            org_batch = [latents, batch[1], None, None]
        else:
            org_batch = [latents, None, None, None]
        print(f"Original batch shape: {org_batch[0].shape}")
        
        # Sample using ddim
        sample_fn = self.diffusion.ddim_sample_loop_known
        
        # Generate samples in latent space
        samples, x_noisy, org = sample_fn(
            model_fn,
            (batch_size, self.vae.latent_dims[0], 64, 64),  # shape
            org_batch,
            org=org_batch,
            clip_denoised=True,
            model_kwargs=cond,
            cond_fn=cond_fn,
            device=latents.device,
            noise_level=noise_level,
        )
        
        # Decode from latent space to image space
        with torch.no_grad():
            vq_samples, _ = self.vae.vq(samples.float())
            generated_images = self.vae.decode(vq_samples)
        
        # Normalize images to [0, 1] range for visualization
        def normalize(img):
            _min = img.min()
            _max = img.max()
            normalized_img = (img - _min)/ (_max - _min)
            return normalized_img
        original_images = normalize(original_images)
        generated_images = normalize(generated_images)
        
        # Calculate difftot (absolute difference between original and generated images)
        difftot = torch.abs(original_images - generated_images).mean(dim=1, keepdim=True)
        
        # Convert difftot to numpy for thresholding
        difftot_np = difftot.squeeze().cpu().numpy()
        print(f"Difftot shape: {difftot_np.shape}")
        
        difftot_masks = []
        
        if batch_size == 1:
            img_for_thresh = (difftot_np * 255).astype(np.uint8)
            img_for_thresh = morphology_close(img_for_thresh)
            ret, thresh1 = cv2.threshold(img_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # thresh1 = remove_small_objects(thresh1.astype(bool), min_size=50, connectivity=5).astype(np.uint8) * 255
            print(f"Otsu threshold value: {ret}")
            difftot_masks.append(thresh1)
        else:
            # Loop through each image in the batch
            for i in range(batch_size): 
                img_for_thresh = (difftot_np[i] * 255).astype(np.uint8) 
                img_for_thresh = morphology_close(img_for_thresh)
                ret, thresh1 = cv2.threshold(img_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # thresh1 = remove_small_objects(thresh1.astype(bool), min_size=50, connectivity=5).astype(np.uint8) * 255
                # print(f"Image {i} Otsu threshold value: {ret}")
                difftot_masks.append(thresh1)

        # Stack all masks into a single tensor with shape (batch_size, height, width)
        difftot_mask = np.array(difftot_masks)
        print(f"Difftot mask shape: {difftot_mask.shape}")

        return {
            "latents": latents,
            "generated_images": generated_images,
            "difftot": difftot_mask,
        }

if __name__ == "__main__":
    import argparse
    import hydra
    from omegaconf import DictConfig
    
    root = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "diffusion")
    
    @hydra.main(version_base=None, config_path=config_path, config_name="latent_diffusion_module.yaml")
    def main(cfg: DictConfig):
        pass