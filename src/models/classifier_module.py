from typing import Any, Dict, Tuple, Optional, List

import os
import wandb
import torch
import pytorch_lightning as pl
from torch import Tensor
from torchmetrics import MeanMetric, Accuracy, Precision, Recall, F1Score
import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.up_down import Encoder, Decoder
from src.models.vae.net import BaseVAE, VQVAE
from src.models.guided_diffusion import dist_util
from src.models.guided_diffusion.resample import create_named_schedule_sampler
from src.models.guided_diffusion.script_util import (
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
    args_to_dict,
)

class ClassifierModule(pl.LightningModule):
    def __init__(
        self,
        vae: Optional[BaseVAE] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        num_timesteps: int = 1000,
        classifier_path: str = None,
        encoder_path: str = None,
        decoder_path: str = None,
        vq_layer_path: str = None,
        use_latent: bool = True,
    ) -> None:
        """Initialize the Classifier Module.

        Args:
            vae (Optional[BaseVAE]): The VAE model used for encoding to latent space.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            num_timesteps (int): Number of diffusion timesteps. Defaults to 1000.
            classifier_path (str): Path to save/load classifier weights. Defaults to None.
            encoder_path (str): Path to the encoder weights. Defaults to None.
            decoder_path (str): Path to the decoder weights. Defaults to None.
            vq_layer_path (str): Path to the VQ layer weights. Defaults to None.
            use_latent (bool): Whether to use latent space for classification. Defaults to True.
            use_fp16 (bool): Whether to use mixed precision. Defaults to False.
            class_cond (bool): Whether to use class conditioning. Defaults to True.
            dataset (str): Dataset name. Defaults to "brats".
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["vae"])
        
        # Initialize metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # Accuracy metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)
        
        # Additional metrics for binary classification
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        
        # VAE for latent space representation
        self.vae = vae
        self.use_latent = use_latent
        
        if self.use_latent and self.vae is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device for VAE: {device}")
            # Load pre-trained VAE weights if paths are provided
            if encoder_path and os.path.exists(encoder_path):
                self.vae.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
                print(f"Loaded encoder weights from {encoder_path}")
            
            if decoder_path and os.path.exists(decoder_path):
                self.vae.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
                print(f"Loaded decoder weights from {decoder_path}")
            
            if hasattr(self.vae, 'vq_layer') and vq_layer_path and os.path.exists(vq_layer_path):
                self.vae.vq_layer.load_state_dict(torch.load(vq_layer_path, map_location=device))
                print(f"Loaded VQ layer weights from {vq_layer_path}")
                
            # Set VAE to evaluation mode and freeze its parameters
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
        
        # Create classifier model and diffusion
        args_dict = classifier_and_diffusion_defaults()
        args_dict.update({"image_size": 64})
        args_dict.update({"diffusion_steps": self.hparams.num_timesteps})
        
        self.classifier, self.diffusion = create_classifier_and_diffusion(
            **args_dict
        )
        
        # # Load pre-trained classifier weights if path is provided
        # if classifier_path and os.path.exists(classifier_path):
        #     self.classifier.load_state_dict(torch.load(classifier_path))
        #     print(f"Loaded classifier weights from {classifier_path}")
        
        # Create schedule sampler
        self.schedule_sampler = create_named_schedule_sampler(
            "uniform", self.diffusion, maxt=num_timesteps
        )
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Best validation accuracy for model saving
        self.best_val_acc = 0.0
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through classifier with noise scheduling.

        Args:
            x (Tensor): Input tensor (either image or latent representation)

        Returns:
            Tensor: Classification logits
        """
        # Sample timesteps
        t, _ = self.schedule_sampler.sample(x.shape[0], dist_util.dev())
        
        # Add noise according to diffusion schedule
        noisy_x = self.diffusion.q_sample(x, t)
        
        # Run through classifier
        return self.classifier(noisy_x, t)
    
    def on_train_epoch_start(self) -> None:
        """Lightning hook called when training begins."""
        # Reset metrics before training starts
        self.train_loss.reset()
        self.val_loss.reset()
        self.train_acc.reset()
        self.val_acc.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
    
    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step.

        Args:
            batch (Tuple): Batch containing images, conditions, masks, and labels
            batch_idx (int): Batch index

        Returns:
            Tensor: Loss value
        """
        imgs, _, _, labels = batch
        
        # Convert to latent representation if using VAE
        if self.use_latent and self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(imgs.float())
                # Handle double_z: if tuple (mu, logvar), take mu
                if isinstance(latents, tuple):
                    imgs = latents[0]  # mu
                else:
                    imgs = latents
        
        # Forward pass
        logits = self(imgs)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        # Update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, labels)
        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch (Tuple): Batch containing images, conditions, masks, and labels
            batch_idx (int): Batch index
        """
        imgs, _, _, labels = batch
        
        # Convert to latent representation if using VAE
        if self.use_latent and self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(imgs.float())
                # Handle double_z: if tuple (mu, logvar), take mu
                if isinstance(latents, tuple):
                    imgs = latents[0]  # mu
                else:
                    imgs = latents
        
        # Forward pass
        logits = self(imgs)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.f1(preds, labels)
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.recall, on_step=False, on_epoch=True)
        self.log("val/f1", self.f1, on_step=False, on_epoch=True)
        
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        wandb.log({
            "Train Accuracy": self.train_acc.compute(),
            "Train Loss": self.train_loss.compute(),
        })
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        current_acc = self.val_acc.compute()
        
        # Save model if validation accuracy improves
        if current_acc > self.best_val_acc:
            self.best_val_acc = current_acc
            print(f"New best validation accuracy: {current_acc:.4f}")
            
            if self.hparams.classifier_path:
                torch.save(
                    self.classifier.state_dict(), 
                    self.hparams.classifier_path
                )
                print(f"Saved model to {self.hparams.classifier_path}")
        
        # Log metrics to wandb if available
        if wandb.run is not None:
            wandb.log({
                "Val Accuracy": self.best_val_acc,
                "Val Loss": self.val_loss.compute(),
                "Val Precision": self.precision.compute(),
                "Val Recall": self.recall.compute(),
                "Val F1": self.f1.compute(),
            })
    
    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """Test step.

        Args:
            batch (Tuple): Batch containing images, conditions, masks, and labels
            batch_idx (int): Batch index
        """
        imgs, _, _, labels = batch
        
        # Convert to latent representation if using VAE
        if self.use_latent and self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(imgs.float())
                # Handle double_z: if tuple (mu, logvar), take mu
                if isinstance(latents, tuple):
                    imgs = latents[0]  # mu
                else:
                    imgs = latents
        
        # Forward pass
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        self.test_loss(loss)
        self.test_acc(preds, labels)
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Prediction step.

        Args:
            batch (Any): Input batch
            batch_idx (int): Batch index
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            Any: Predictions
        """
        if isinstance(batch, Tensor):
            imgs = batch
        else:
            imgs = batch[0]
        
        # Convert to latent representation if using VAE
        if self.use_latent and self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(imgs.float())
                # Handle double_z: if tuple (mu, logvar), take mu
                if isinstance(latents, tuple):
                    imgs = latents[0]  # mu
                else:
                    imgs = latents
        
        # Forward pass
        logits = self(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        return {"logits": logits, "probs": probs, "preds": preds}
    
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

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    
    root = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model")
    
    @hydra.main(version_base=None, config_path=config_path, config_name="classifier_module.yaml")
    def main(cfg: DictConfig):
        # Create classifier module
        classifier_module = hydra.utils.instantiate(cfg)
        
        # Test forward pass
        batch_size = 2
        channels = 4 if cfg.dataset == "brats" else 1
        height, width = 64, 64
        
        x = torch.randn(batch_size, channels, height, width)
        labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        logits = classifier_module(x)
        
        print("Input shape:", x.shape)
        print("Output logits shape:", logits.shape)
    
    main()