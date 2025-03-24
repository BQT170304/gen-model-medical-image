from typing import Any, Dict, Tuple, Optional, List

import os
import torch
import pytorch_lightning as pl
from torch import Tensor
from torchmetrics import MeanMetric
import numpy as np
import cv2
from skimage.morphology import remove_small_objects
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.up_down import Encoder, Decoder
from src.models.vae.net import BaseVAE, VQVAE
from src.utils.ema import LitEma
from src.models.diffusion.guided_diffusion import dist_util
from src.models.diffusion.guided_diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from src.models.diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


class LatentDiffusionModule(pl.LightningModule):
    def __init__(
        self,
        vae: BaseVAE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        encoder_path: str,
        decoder_path: str,
        vq_layer_path: str,
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
        
        # Load VAE components
        self.vae = vae
        
        # Load pre-trained weights if paths are provided
        # if os.path.exists(encoder_path):
        #     self.vae.encoder.load_state_dict(torch.load(encoder_path))
        
        # if os.path.exists(decoder_path):
        #     self.vae.decoder.load_state_dict(torch.load(decoder_path))
        
        # if os.path.exists(vq_layer_path):
        #     self.vae.vq_layer.load_state_dict(torch.load(vq_layer_path))
            
        self.vae.eval()
        # Freeze VAE 
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # Create diffusion model
        defaults = self.get_default_args()
        args_dict = model_and_diffusion_defaults()
        args_dict.update(defaults)
        # print("args_dict: ", args_dict)
        self.model, self.diffusion = create_model_and_diffusion(
            **args_dict
        )
        
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
    
    def get_default_args(self):
        """Return default arguments for the diffusion model."""
        defaults = dict(
            use_fp16=False,
            dataset='brats',
            class_cond=False,
            image_size=32,
            num_channels=64,
            num_res_blocks=2,
            attention_resolutions="16,8",
            dropout=0.0,
            learn_sigma=False,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing="ddim1000",
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
            latents, _ = self.vae.encode(imgs.float())
            
            # Normalize latents to [-1, 1]
            _max = latents.max()
            _min = latents.min()
            latents = (latents - _min) / (_max - _min)  # [0, 1]
            latents = 2 * latents - 1  # [-1, 1]
        
        # Apply diffusion model on latents
        # loss = self(latents, cond)
        loss = self(latents, None) # Khong can condition
        
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
            latents, _ = self.vae.encode(imgs.float())
            
            # Normalize latents to [-1, 1]
            _max = latents.max()
            _min = latents.min()
            latents = (latents - _min) / (_max - _min)  # [0, 1]
            latents = 2 * latents - 1  # [-1, 1]
        
        # Apply diffusion model on latents
        # loss = self(latents, cond)
        loss = self(latents, None) # Khong can condition
        
        # Update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self.best_val_loss >= self.val_loss.compute():
            self.best_val_loss = self.val_loss.compute()
            print("SAVED!")
            torch.save(
                self.model.state_dict(), 
                f"/data/hpc/qtung/gen-model-boilerplate/src/ckpt/latent_diffusion/diffusion.pth"
            )
    
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
    
    def sample(self, batch, cond: Dict, batch_size: int = 1) -> Dict[str, Tensor]:
        """Generate samples using the diffusion model and calculate difftot as a binary mask.

        Args:
            cond (Dict): Conditioning information.
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            Dict[str, Tensor]: Dictionary containing original images, generated images, and difftot mask.
        """
        # Generate samples in latent space
        original_images = batch[0]
        
        # Encode original images to latent space
        with torch.no_grad():
            latents, _ = self.vae.encode(original_images.float())
            
            # Normalize latents to [-1, 1]
            _max = latents.max()
            _min = latents.min()
            latents = (latents - _min) / (_max - _min)  # [0, 1]
            latents = 2 * latents - 1  # [-1, 1]
        
        # Setup model function for diffusion sampling
        def model_fn(x, t, y=None):
            return self.model(x, t, y if cond.get("y") is not None else None)
        
        # Setup condition function (can be None or implemented if needed)
        cond_fn = None
        
        # Create a batch of original data in the format expected by ddim_sample_loop_known
        org_batch = [latents, cond.get("y", None), None, None]
        
        # Sample using ddim
        sample_fn = self.diffusion.ddim_sample_loop_known
        
        # Generate samples in latent space
        samples, x_noisy, org = sample_fn(
            model_fn,
            (batch_size, self.vae.latent_dims[0], 32, 32),  # shape
            org_batch,
            org=org_batch,
            clip_denoised=True,
            model_kwargs=cond,
            cond_fn=cond_fn,
            device=latents.device,
            noise_level=500  # Adjustable parameter
        )
        
        # Denormalize samples if normalized during encoding
        samples = ((samples + 1) / 2) * (_max - _min) + _min
        
        # Decode from latent space to image space
        with torch.no_grad():
            generated_images = self.vae.decode(samples)
        
        # Normalize images to [0, 1] range for visualization
        generated_images = (generated_images + 1) / 2
        original_images = (original_images + 1) / 2
        
        # Calculate difftot (absolute difference between original and generated images)
        difftot = torch.abs(original_images - generated_images).mean(dim=1, keepdim=True)
        
        # Convert difftot to numpy for thresholding
        difftot_np = difftot.squeeze().cpu().numpy()
        print(f"Difftot shape: {difftot_np.shape}")
        
        # Apply Otsu thresholding
        img_for_thresh = (difftot_np * 255).astype(np.uint8)  # Scale to 0-255 for thresholding
        ret, thresh1 = cv2.threshold(img_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Otsu threshold value: {ret}")
        
        # # Apply mask cleaning to remove small objects
        # min_size = 0  # Adjust this value as needed
        # mask_cleaned = remove_small_objects(thresh1.astype(bool), min_size=min_size)
        
        # Convert threshold result back to torch tensor
        # Check number of dimensions and add one if needed
        if thresh1.ndim == 2:  # If 2D (height, width)
            thresh1 = np.expand_dims(thresh1, axis=0)  # Add channel dimension (1, height, width)
            
        difftot_mask = thresh1
        
        # Return results as a dictionary
        return {
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