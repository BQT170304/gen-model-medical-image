import os
import cv2
import wandb
import numpy as np
import functools
import argparse
import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import Logger
import hydra
from omegaconf import DictConfig
import rootutils
from skimage.morphology import remove_small_objects
import logging
from torchvision.datasets import CIFAR10
from torchvision import transforms as tf
from models.guided_diffusion.resample import LossAwareSampler, UniformSampler
import torchmetrics as tm
from models.guided_diffusion.bratsloader import BRATSDataset
from models.guided_diffusion import dist_util, logger
from models.guided_diffusion.image_datasets import load_data
from models.guided_diffusion.resample import create_named_schedule_sampler
from models.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from models.guided_diffusion.train_util import TrainLoop

wandb.login(key="a7df3964ffcf4c24262d52ad7ea0ee7a362b8fbc")
# Initialize wandb
wandb.init(project='vnu-uet-2004', name="Diffusion")

logging.basicConfig(level=logging.INFO)

# import from src after this line
root_path = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True)

class UnetDiffusionModule(L.LightningModule):
    def __init__( self, args=None, 
                        diffusion_path:str = str(root_path / "src" / "diffusion" / "openai_unet_diffusion_cross4.pth"),
                        num_timesteps: int = 1000 ):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.top_1 = []
        self.top_2 = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.correct = 0
        self.total = 0
        self.Dice = []
        self.IOU = []
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        # self.model = torch.load('/home/tqlong/minhdd/anomaly/src/diffusion/openai_unet_diffusion_dice_0.2541_iou_0.0863.pth')

        self.criterion = self.diffusion.training_losses

        self.diff_test = None
        self.class_test = None

        # self.criterion = FocalLoss()
        # self.criterion = nn.CrossEntropyLoss()

        self.diffusion_path = diffusion_path
        # Best Loss
        self.best_eval_dice = float('-inf')
        self.schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, self.diffusion, maxt=num_timesteps
        )

        # logger.log("loading classifier...")
        self.class_test = torch.load('/home/tqlong/minhdd/anomaly/src/classifier/openai_unet_classifier_cross4.pth')
        self.class_test.to(dist_util.dev())
        self.class_test.eval()
        

    def forward(self, imgs, cond):
        t, weights = self.schedule_sampler.sample(imgs.shape[0], dist_util.dev())
        imgs.half()
        # print(imgs.shape)
        # print(t.shape)
        t.half()
        # self.compute_losses = functools.partial(
        #         self.diffusion.training_losses,
        #         self.model,
        #         imgs,
        #         t,
        #         model_kwargs=cond,
        #     )

        losses1 = self.criterion(self.model, imgs, t, model_kwargs=cond)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        losses = losses1[0]
        sample = losses1[1]

        loss = (losses["loss"] * weights).mean()

        lossmse = (losses["mse"] * weights).mean().detach()

        return loss

    
    def training_step(self, batch):
        imgs, cond, _, labels = batch
        loss = self(imgs, cond)

        # if loss > 10.0 and self.current_epoch > 2:
        #     print(loss)
        self.train_loss.append(loss.item())
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, cond, mask, labels = batch
        
        loss = self(imgs, cond)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.val_loss.append(loss.item())

        if (self.current_epoch + 1) % 10 != 0:
            return
        ### Sample
        print(f"labels: {labels}")

        def cond_fn(x, t,  y=None):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = self.class_test(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                a=torch.autograd.grad(selected.sum(), x_in)[0]
                return  a, a * 100



        def model_fn(x, t, y=None):
            assert y is not None
            return self.model(x, t, y)

        mask = mask.squeeze().cpu().numpy()  # squeeze() sẽ loại bỏ các chiều có size = 1
        print(f"mask shape: {mask.shape}")
        model_kwargs = {}

        classes = torch.randint(
            low=0, high=1, size=(batch[0].shape[0],), device=dist_util.dev()
        )
        model_kwargs["y"] = classes

        print('y', model_kwargs["y"])

        sample_fn = (
            self.diffusion.ddim_sample_loop_known
        )
        # print('samplefn', sample_fn)
        sample, x_noisy, org = sample_fn(
            model_fn,
            (batch[0].shape[0], 4, 128, 128), batch, org=batch,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=500
        )

        org = (org + 1) / 2
        sample = (sample + 1) / 2
        dice = 0.0
        iou = 0.0

        for i in range(batch[0].shape[0]):

            difftot=(org[i, :4,...]-sample[i, ...]).mean(dim=0)
            difftot = torch.clamp(difftot, min=0)
            difftot = difftot.cpu().numpy()

            # Convert to 8-bit (values 0-255)
            img_for_thresh = (difftot * 255).astype(np.uint8)

            # Now apply Otsu thresholding
            ret, thresh1 = cv2.threshold(img_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"mask shape: {mask.shape}")
            print(f"thresh1 shape: {thresh1.shape}")

            dice += dice_score(mask[i], thresh1)
            iou += iou_score(mask[i], thresh1)

        # if dice < 5.0:
        #     min_size = 3
        #     mask_cleaned = remove_small_objects(thresh1, min_size=min_size)

        #     # Bước 3: Áp dụng mask đã lọc để loại bỏ vùng nhỏ khỏi difftot
        #     # Ở đây, các vùng không đạt yêu cầu (mask_cleaned == False) được gán giá trị 0
        #     thresh1 = thresh1 * mask_cleaned
        
        # dice = dice_score(mask, thresh1)
        # iou = iou_score(mask, thresh1)

        self.Dice.append(dice.item() / batch[0].shape[0])
        self.IOU.append(iou.item() / batch[0].shape[0])


    
    def on_validation_epoch_end(self):
        # Mean Losses and Acc
        mean_val_loss = sum(self.val_loss) / len(self.val_loss)
        mean_train_loss = sum(self.train_loss) / len(self.train_loss) if len(self.train_loss) !=0  else 0
        mean_dice = sum(self.Dice) / len(self.Dice) if len(self.Dice) !=0  else 0
        mean_iou = sum(self.IOU) / len(self.IOU) if len(self.IOU) !=0  else 0

        # reset 
        self.val_loss = []
        self.Dice = []
        self.IOU = []
        
        # Save based on train-loss
        if mean_dice > self.best_eval_dice:
            self.best_eval_dice = mean_dice
            torch.save(self.model, str(root_path / "src" / "diffusion" / f"openai_unet_diffusion_dice_{self.best_eval_dice:.4f}_iou_{mean_iou:.4f}.pth"))      
        
        # Display
        print()
        print(f'{self.current_epoch+1:<6} train loss: {mean_train_loss:<10.4f}, val loss: {mean_val_loss:<10.4f}, dice: {mean_dice:<10.4f}, iou: {mean_iou:<10.4f},  best dice score: {self.best_eval_dice:<10.4f}', end = "   ")
        print()

        # Log metrics to wandb
        wandb.log({
            "Train loss": mean_train_loss,
            "Val Loss": mean_val_loss,
            "Dice score": mean_dice,
            "IOU score": mean_iou,
            "Best Dice": self.best_eval_dice,
        })
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img.cpu().numpy()

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

def main():
    args = create_argparser().parse_args()
    model = UnetDiffusionModule(args=args)

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model = model.to(device)

    # Hyperparameters for testing
    batch_size = 1
    num_classes = 10
    img_size = (64, 64, 4)
    

    # Create random data for testing and move to the device
    random_imgs = torch.randn(img_size).float().to(device)
    random_imgs = random_imgs.permute(2, 0, 1)  # Random images (batch, channels, height, width)
    # Thêm batch dimension để đầu vào có kích thước [batch_size, channels, height, width]
    random_imgs = random_imgs.unsqueeze(0).to(device)



    # Forward pass through the model
    out = model(random_imgs)
    
    # Print the output shape
    print(out)
    print(out.shape)
    print("END")

if __name__ == '__main__':
    main()