from typing import Any, Tuple, List
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import random
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio)
from torchmetrics.image.fid import FrechetInceptionDistance  
from torchmetrics.image.inception import InceptionScore

from torchmetrics import JaccardIndex, MeanMetric
from torchmetrics.segmentation import DiceScore as Dice

from src.models.gan import GANModule, CGANModule
from src.models.diffusion import DiffusionModule, ConditionDiffusionModule, LatentDiffusionModule
from src.models.vae import VAEModule
from src.models.unet import UNetModule
from src.models.flow import NFModule


class Metrics(Callback):

    def __init__(
        self,
        metric_list: List[str],
        mean: float = 0.5,
        std: float = 0.5,
        n_ensemble: int = 1,
        max_batches_to_keep: int = 50,  # Maximum number of batches to collect
    ) -> None:
        """_summary_

        Args:
            metric_list (List[str]): _description_
            mean (float, optional): to convert image into (0, 1). Defaults to 0.5.
            std (float, optional): to convert image into (0, 1). Defaults to 0.5.
            n_ensemble (int, optional): for segmentation with diffusion model. Defaults to 1.
            max_batches_to_keep (int, optional): max number of batches to collect per epoch. Defaults to 50.
        """

        for metric in metric_list:
            if metric == "binary-dice":
                self.train_dice = Dice(ignore_index=0)
                self.val_dice = Dice(ignore_index=0)
                self.test_dice = Dice(ignore_index=0)

            elif metric == "binary-iou":
                self.train_iou = JaccardIndex(task="binary")
                self.val_iou = JaccardIndex(task="binary")
                self.test_iou = JaccardIndex(task="binary")

            elif metric == "ssim":
                self.train_ssim = StructuralSimilarityIndexMeasure(data_range=2.)
                self.val_ssim = StructuralSimilarityIndexMeasure(data_range=2.)
                self.test_ssim = StructuralSimilarityIndexMeasure(data_range=2.)

            elif metric == "psnr":
                self.train_psnr = PeakSignalNoiseRatio(data_range=2.)
                self.val_psnr = PeakSignalNoiseRatio(data_range=2.)
                self.test_psnr = PeakSignalNoiseRatio(data_range=2.)

            # Check: GPU if use train,val,test. save memory
            elif metric == "fid":
                self.val_fid = FrechetInceptionDistance(normalize=True)
                self.test_fid = FrechetInceptionDistance(normalize=True)

            elif metric == "is":
                self.val_is = InceptionScore(normalize=True)
                self.test_is = InceptionScore(normalize=True)

            elif metric == "image_variance":
                self.train_image_variance = MeanMetric()
                self.val_image_variance = MeanMetric()
                self.test_image_variance = MeanMetric()
            
            elif metric == "boundary_variance":
                self.train_boundary_variance = MeanMetric()
                self.val_boundary_variance = MeanMetric()
                self.test_boundary_variance = MeanMetric()

            else:
                NotImplementedError(f"Not implemented for {metric} metric")

        self.metric_list = metric_list
        self.mean = mean
        self.std = std
        self.n_ensemble = n_ensemble
        self.max_batches_to_keep = max_batches_to_keep
        
        # Initialize batch collection lists
        self.train_batches = []
        self.val_batches = []
        self.test_batches = []

    def metrics2device(self, device):
        if "binary-dice" in self.metric_list:
            self.train_dice.to(device)
            self.val_dice.to(device)
            self.test_dice.to(device)

        if "binary-iou" in self.metric_list:
            self.train_iou.to(device)
            self.val_iou.to(device)
            self.test_iou.to(device)

        if "ssim" in self.metric_list:
            self.train_ssim.to(device)
            self.val_ssim.to(device)
            self.test_ssim.to(device)

        if "psnr" in self.metric_list:
            self.train_psnr.to(device)
            self.val_psnr.to(device)
            self.test_psnr.to(device)

        if "image_variance" in self.metric_list:
            self.train_image_variance.to(device)
            self.val_image_variance.to(device)
            self.test_image_variance.to(device)

        if "boundary_variance" in self.metric_list:
            self.train_boundary_variance.to(device)
            self.val_boundary_variance.to(device)
            self.test_boundary_variance.to(device)

        if "fid" in self.metric_list:
            self.val_fid.to(device)
            self.test_fid.to(device)

        if "is" in self.metric_list:
            self.val_is.to(device)
            self.test_is.to(device)

    def reset_metrics(self):
        if "binary-dice" in self.metric_list:
            self.val_dice.reset()

        if "binary-iou" in self.metric_list:
            self.val_iou.reset()

        if "ssim" in self.metric_list:
            self.val_ssim.reset()

        if "psnr" in self.metric_list:
            self.val_psnr.reset()

        if "image_variance" in self.metric_list:
            self.val_image_variance.reset()

        if "boundary_variance" in self.metric_list:
            self.val_boundary_variance.reset()

        if "fid" in self.metric_list:
            self.val_fid.reset()

        if "is" in self.metric_list:
            self.val_is.reset()
        
        # Also reset batch collections
        self.train_batches = []
        self.val_batches = []
        self.test_batches = []

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.reset_metrics()
        self.metrics2device(pl_module.device)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                        outputs: STEP_OUTPUT, batch: Any,
                        batch_idx: int) -> None:
        # Collect batch for epoch-end processing
        if len(self.train_batches) < self.max_batches_to_keep:
            self.train_batches.append(batch)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics2device(pl_module.device)

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: STEP_OUTPUT | None, batch: Any,
                                batch_idx: int) -> None:
        # Collect batch for epoch-end processing
        if len(self.val_batches) < self.max_batches_to_keep:
            self.val_batches.append(batch)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics2device(pl_module.device)

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                        outputs: STEP_OUTPUT | None, batch: Any,
                        batch_idx: int) -> None:
        # Collect batch for epoch-end processing
        if len(self.test_batches) < self.max_batches_to_keep:
            self.test_batches.append(batch)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute metrics at the end of the training epoch."""
        if not self.train_batches:
            return
            
        # Randomly select a batch to evaluate
        batch = random.choice(self.train_batches)
        
        # Get predictions and targets
        preds, targets = self.infer(pl_module, batch)  # range [0, 1]
        
        # Calculate and log image variance metrics if needed
        if "image_variance" in self.metric_list or "boundary_variance" in self.metric_list:
            # (b, n, c, w, h) -> (b, c, w, h)
            variance = ((preds > 0.5).to(torch.float32)).var(dim=1)

            if "image_variance" in self.metric_list:
                self.train_image_variance.update(variance.mean())
                pl_module.log("train/image_variance",
                            self.train_image_variance,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                            metric_attribute="train_image_variance")

            if "boundary_variance" in self.metric_list:
                boundary = variance > 0
                boundary_variance = ((variance).sum(dim=[1, 2, 3]) + \
                                    1) / (boundary.sum(dim=[1, 2, 3]) + 1)
                self.train_boundary_variance.update(boundary_variance)
                pl_module.log("train/boundary_variance",
                            self.train_boundary_variance,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                            metric_attribute="train_boundary_variance")
                            
        # Process ensemble predictions if needed
        if len(preds.shape) == 5: # [b, n, c, w, h]
            preds = preds.mean(dim=1)

        # Calculate and log SSIM if needed
        if "ssim" in self.metric_list:
            self.train_ssim(preds, targets)
            pl_module.log("train/ssim",
                        self.train_ssim,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="train_ssim")

        # Calculate and log PSNR if needed
        if "psnr" in self.metric_list:
            self.train_psnr(preds, targets)
            pl_module.log("train/psnr",
                        self.train_psnr,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="train_psnr")

        # Calculate and log segmentation metrics if needed
        if "binary-dice" in self.metric_list or "binary-iou" in self.metric_list:
            preds = preds.to(torch.int64)
            targets = targets.to(torch.int64)

            if "binary-dice" in self.metric_list:
                self.train_dice(preds, targets)
                pl_module.log("train/dice",
                        self.train_dice,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="train_dice")

            if "binary-iou" in self.metric_list:
                self.train_iou(preds, targets)
                pl_module.log("train/iou",
                        self.train_iou,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="train_iou")
        
        # Clear collected batches after processing
        self.train_batches = []

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute metrics at the end of the validation epoch."""
        if not self.val_batches:
            return
            
        # Randomly select a batch to evaluate
        batch = random.choice(self.val_batches)
        
        # Get predictions and targets
        preds, targets = self.infer(pl_module, batch)  # range [0, 1]
        
        # Calculate and log image variance metrics if needed
        if "image_variance" in self.metric_list or "boundary_variance" in self.metric_list:
            # (b, n, c, w, h) -> (b, c, w, h)
            variance = ((preds > 0.5).to(torch.float32)).var(dim=1)

            if "image_variance" in self.metric_list:
                self.val_image_variance.update(variance.mean())
                pl_module.log("val/image_variance",
                            self.val_image_variance,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                            metric_attribute="val_image_variance")

            if "boundary_variance" in self.metric_list:
                boundary = variance > 0
                boundary_variance = ((variance).sum(dim=[1, 2, 3]) + \
                                    1) / (boundary.sum(dim=[1, 2, 3]) + 1)
                self.val_boundary_variance.update(boundary_variance)
                pl_module.log("val/boundary_variance",
                            self.val_boundary_variance,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                            metric_attribute="val_boundary_variance")
                            
        # Process ensemble predictions if needed
        if len(preds.shape) == 5: # [b, n, c, w, h]
            preds = preds.mean(dim=1)

        # Calculate and log SSIM if needed
        if "ssim" in self.metric_list:
            self.val_ssim(preds, targets)
            pl_module.log("val/ssim",
                        self.val_ssim,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="val_ssim")

        # Calculate and log PSNR if needed
        if "psnr" in self.metric_list:
            self.val_psnr(preds, targets)
            pl_module.log("val/psnr",
                        self.val_psnr,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="val_psnr")

        # Calculate and log segmentation metrics if needed
        if "binary-dice" in self.metric_list or "binary-iou" in self.metric_list:
            preds = preds.to(torch.int64)
            targets = targets.to(torch.int64)

            if "binary-dice" in self.metric_list:
                self.val_dice(preds, targets)
                pl_module.log("val/dice",
                        self.val_dice,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="val_dice")

            if "binary-iou" in self.metric_list:
                self.val_iou(preds, targets)
                pl_module.log("val/iou",
                        self.val_iou,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="val_iou")

        # Calculate and log GAN-specific metrics if needed
        if "fid" in self.metric_list or "is" in self.metric_list:
            fakes, reals = preds, targets

            if preds.shape[1] == 1:
                # gray to rgb image
                fakes = torch.cat([fakes, fakes, fakes], dim=1)
                reals = torch.cat([reals, reals, reals], dim=1)

            reals = torch.nn.functional.interpolate(reals,
                                                    size=(299, 299),
                                                    mode="bilinear")
            fakes = torch.nn.functional.interpolate(fakes,
                                                    size=(299, 299),
                                                    mode="bilinear")

            if "fid" in self.metric_list:
                self.val_fid.update(reals, real=True)
                self.val_fid.update(fakes, real=False)
                pl_module.log("val/fid",
                        self.val_fid,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="val_fid")

            if "is" in self.metric_list:
                self.val_is.update(fakes)
                score, std = self.val_is.compute()
                pl_module.log("val/is", score, metric_attribute="val_is")
                self.val_is.reset()
        
        # Clear collected batches after processing
        self.val_batches = []

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute metrics at the end of the test epoch."""
        if not self.test_batches:
            return
            
        # Randomly select a batch to evaluate
        batch = random.choice(self.test_batches)
        
        # Get predictions and targets
        preds, targets = self.infer(pl_module, batch)  # range [0, 1]
        
        # Calculate and log image variance metrics if needed
        if "image_variance" in self.metric_list or "boundary_variance" in self.metric_list:
            # (b, n, c, w, h) -> (b, c, w, h)
            variance = ((preds > 0.5).to(torch.float32)).var(dim=1)

            if "image_variance" in self.metric_list:
                self.test_image_variance.update(variance.mean())
                pl_module.log("test/image_variance",
                            self.test_image_variance,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                            metric_attribute="test_image_variance")

            if "boundary_variance" in self.metric_list:
                boundary = variance > 0
                boundary_variance = ((variance).sum(dim=[1, 2, 3]) + \
                                    1) / (boundary.sum(dim=[1, 2, 3]) + 1)
                self.test_boundary_variance.update(boundary_variance)
                pl_module.log("test/boundary_variance",
                            self.test_boundary_variance,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                            metric_attribute="test_boundary_variance")
                            
        # Process ensemble predictions if needed
        if len(preds.shape) == 5: # [b, n, c, w, h]
            preds = preds.mean(dim=1)

        # Calculate and log SSIM if needed
        if "ssim" in self.metric_list:
            self.test_ssim(preds, targets)
            pl_module.log("test/ssim",
                        self.test_ssim,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="test_ssim")

        # Calculate and log PSNR if needed
        if "psnr" in self.metric_list:
            self.test_psnr(preds, targets)
            pl_module.log("test/psnr",
                        self.test_psnr,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="test_psnr")

        # Calculate and log segmentation metrics if needed
        if "binary-dice" in self.metric_list or "binary-iou" in self.metric_list:
            preds = preds.to(torch.int64)
            targets = targets.to(torch.int64)

            if "binary-dice" in self.metric_list:
                self.test_dice(preds, targets)
                pl_module.log("test/dice",
                        self.test_dice,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="test_dice")

            if "binary-iou" in self.metric_list:
                self.test_iou(preds, targets)
                pl_module.log("test/iou",
                        self.test_iou,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="test_iou")

        # Calculate and log GAN-specific metrics if needed
        if "fid" in self.metric_list or "is" in self.metric_list:
            fakes, reals = preds, targets

            if preds.shape[1] == 1:
                # gray to rgb image
                fakes = torch.cat([fakes, fakes, fakes], dim=1)
                reals = torch.cat([reals, reals, reals], dim=1)

            reals = torch.nn.functional.interpolate(reals,
                                                    size=(299, 299),
                                                    mode="bilinear")
            fakes = torch.nn.functional.interpolate(fakes,
                                                    size=(299, 299),
                                                    mode="bilinear")

            if "fid" in self.metric_list:
                self.test_fid.update(reals, real=True)
                self.test_fid.update(fakes, real=False)
                pl_module.log("test/fid",
                        self.test_fid,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        metric_attribute="test_fid")

            if "is" in self.metric_list:
                self.test_is.update(fakes)
                score, std = self.test_is.compute()
                pl_module.log("test/is", score, metric_attribute="test_is")
                self.test_is.reset()
        
        # Clear collected batches after processing
        self.test_batches = []

    def rescale(self, image: Tensor) -> Tensor:
        #convert range (-1, 1) to (0, 1)
        return (image * self.std + self.mean).clamp(0, 1)

    def infer(self, pl_module: LightningModule, batch: Any) -> Tuple[Tensor, Tensor]:
        
        if isinstance(pl_module, UNetModule):
            preds = pl_module.predict(batch[1]["image"]) # range [0, 1]

        elif isinstance(pl_module, VAEModule):
            preds = pl_module.predict(batch[0]) # range [0, 1]

        elif isinstance(pl_module, DiffusionModule):
            fakes = []
            for _ in range(self.n_ensemble):
                conds=batch[1].copy() if isinstance(pl_module, ConditionDiffusionModule) else None
                samples = pl_module.predict(num_sample=batch[0].shape[0],
                                            device=pl_module.device,
                                            cond=conds) #  range (-1, 1)
                fakes.append(samples[-1])  # [b, c, w, h]
            
            fakes = torch.stack(fakes, dim=1)  # (b, n, c, w, h)
            preds = self.rescale(fakes) # range [0, 1]
            
        elif isinstance(pl_module, LatentDiffusionModule):
            reals = batch[0]
            conds = None
            
            # Chuẩn bị điều kiện nếu cần
            if len(batch) > 1 and isinstance(batch[1], dict):
                conds = {key: value for key, value in batch[1].items()}
            
            # Tạo dictionary conditioning với original_images
            conditioning = {}
            if conds is not None:
                conditioning.update(conds)
            
            # Tạo danh sách để chứa kết quả
            fakes = []
            
            # Lặp qua n_ensemble lần để tạo nhiều mẫu
            for _ in range(self.n_ensemble):
                # Gọi hàm sample của LatentDiffusionModule
                results = pl_module.sample(
                    batch=[reals, conds, None, None],
                    cond=conditioning,
                    batch_size=reals.shape[0],
                )
                
                # Lấy ảnh đã tạo và difftot từ kết quả
                generated_images = results["generated_images"]
                
                fakes.append(generated_images)
            
            # Chuyển các mẫu thành tensor
            fakes = torch.stack(fakes, dim=1)  # b, n, c, w, h
            
            # Tính giá trị trung bình nếu có nhiều mẫu
            if self.n_ensemble > 1:
                # Tính toán variance nếu cần
                self.compute_variance(pl_module, reals, fakes, conds, mode)
            
            # # Lấy giá trị trung bình của các mẫu
            # fakes = fakes.mean(dim=1)  # b, c, w, h
            preds = self.rescale(fakes)
            
        elif isinstance(pl_module, GANModule):
            conds=batch[1] if isinstance(pl_module, CGANModule) else None
            samples = pl_module.predict(cond=conds,
                                        num_sample=batch[0].shape[0],
                                        device=pl_module.device) # range [-1, 1]
            preds = self.rescale(samples) # range [0, 1]

        elif isinstance(pl_module, NFModule):
            fakes = pl_module.predict(num_sample=batch[0].shape[0],
                                    device=pl_module.device) # range [-1, 1]

            preds = self.rescale(fakes) # range [0, 1]

        else:
            raise NotImplementedError("This module is not implemented")

        return preds, self.rescale(batch[0]) # range [0, 1]

