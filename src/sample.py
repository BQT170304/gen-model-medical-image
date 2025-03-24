"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from skimage.morphology import remove_small_objects
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import gaussian_blur
import torch as th
import torch.distributed as dist
from data.dataset.brats2020 import BraTS2020DatasetClassifier
from models.diffusion.guided_diffusion.image_datasets import load_data
from models.diffusion.guided_diffusion import dist_util, logger
from models.components.up_down.encoder import Encoder
from models.components.up_down.decoder import Decoder 
from models.diffusion.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
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


def visualize_and_save(sample, org, number, mask):
    org = (org + 1) / 2
    sample = (sample + 1) / 2
    org = org.cpu()
    sample = sample.cpu()

    # Create a figure and set of subplots
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))  # 1 row, 5 columns

    # Plot the images
    axes[0].imshow(sample[0, 0, ...].numpy(), cmap='gray')  # Display grayscale or use a suitable colormap
    axes[0].set_title('sampled output0')
    # print("sample")
    # print(sample[0, 0, ...].min())
    # print(sample[0, 0, ...].max())

    axes[1].imshow(org[0, 0, ...].numpy(), cmap='gray')
    axes[1].set_title('org')
    # print("org")
    # print(org[0, 0, ...].min())
    # print(org[0, 0, ...].max())

    # axes[2].imshow(visualize(sample[0, 2, ...]), cmap='gray')
    # axes[2].set_title('sampled output2')

    # axes[3].imshow(visualize(sample[0, 3, ...]), cmap='gray')
    # axes[3].set_title('sampled output3')

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('mask')

    # Calculate the difference total (difftot) and plot it
    # Visualize heatmap with customized vmin, vmax
    difftot=(org[0, :4,...]-sample[0, ...]).mean(dim=0)
    difftot = th.clamp(difftot, min=0)
    difftot = difftot.numpy()

    # print(difftot.shape)
    axes[3].imshow(difftot, cmap='plasma')
    axes[3].set_title('difftot')

    # Convert to 8-bit (values 0-255)
    img_for_thresh = (difftot * 255).astype(np.uint8)

    # Now apply Otsu thresholding
    ret, thresh1 = cv2.threshold(img_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"ret: {ret}")
    min_size = 0
    mask_cleaned = remove_small_objects(thresh1.astype(bool), min_size=min_size)

    # Bước 3: Áp dụng mask đã lọc để loại bỏ vùng nhỏ khỏi difftot
    # Ở đây, các vùng không đạt yêu cầu (mask_cleaned == False) được gán giá trị 0
    thresh1 = thresh1 * mask_cleaned

    dice = dice_score(mask, thresh1)
    iou = iou_score(mask, thresh1)
    # Plot the images
    axes[4].imshow(thresh1, cmap='gray')  # Display grayscale or use a suitable colormap
    axes[4].set_title('apply thresold')

    # Add colorbar for better understanding of value ranges
    # plt.colorbar(heatmap, ax=axes[5])

    # Optionally, add contours to highlight the differences
    # contours = axes[5].contour(difftot.cpu().numpy(), levels=5, colors='black', linewidths=0.5)
    # axes[5].clabel(contours, inline=True, fontsize=8)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'/data/hpc/minhdd/anomaly/src/sample_ldm/visualized_images_{number}.png')
    plt.show()

    return dice, iou

def main():
    vae = False
    if vae == True:
        # Load the state dictionaries
        encoder_state_dict = th.load("/data/hpc/minhdd/anomaly/src/vae2/encoder2_0.00006756.pth")
        decoder_state_dict = th.load("/data/hpc/minhdd/anomaly/src/vae2/decoder2_0.00006756.pth")

        # Instantiate the models
        in_channels = 4
        latent_dims = [4, 8, 16]  # Example tuple or list, so latent_dims[0] = 4
        z_channels = latent_dims[0]
        base_channels = 64
        channel_multipliers = [1, 2, 4, 8]

        encoder = Encoder(
            in_channels=in_channels,  # 4
            z_channels=z_channels,    # 4
            base_channels=base_channels,
            block="Residual",
            n_layer_blocks=1,
            drop_rate=0.0,
            channel_multipliers=channel_multipliers,
            attention="Attention",
            n_attention_heads=None,
            n_attention_layers=None,
            double_z=False
        )

        decoder = Decoder(
            out_channels=in_channels,
            z_channels=z_channels,
            base_channels=base_channels,
            block="Residual",
            n_layer_blocks=1,
            drop_rate=0.0,
            channel_multipliers=channel_multipliers,
            attention="Attention",
            n_attention_heads=None,
            n_attention_layers=None
        )

        # Load the weights into the model instances
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)

    number = 0
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
   
    # model = th.load("/data/hpc/minhdd/anomaly/src/diffusion/openai_unet_diffusion_dice_0.6893_iou_0.5552.pth")
    model_state_dict = th.load("/data/hpc/minhdd/anomaly/src/diffusion/openai_unet_diffusion1_dice_0.0023_iou_0.0012.pth")
    model.load_state_dict(model_state_dict)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # encoder.to(dist_util.dev())
    # decoder.to(dist_util.dev())
    # encoder.eval()
    # decoder.eval()

    logger.log("loading classifier...")
    classifier = th.load('/data/hpc/minhdd/anomaly/src/classifier/openai_unet_classifier_cross4.pth')

    
    ds = BraTS2020DatasetClassifier(args.data_dir, mode="test")
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)


    print('loaded classifier')
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)


    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()


    def cond_fn(x, t,  y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            return  a, a * args.classifier_scale



    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # def cond_fn(x, t,  y=None):
    #     assert y is not None
    #     with th.enable_grad():
    #         x_in = x.detach().requires_grad_(True)
    #         logits = classifier(x_in, t)
    #         log_probs = F.log_softmax(logits, dim=-1)
    #         selected = log_probs[range(len(logits)), y.view(-1)]
    #         a=th.autograd.grad(selected.sum(), x_in)[0]
    #         return  a, a * 100



    # def model_fn(x, t, y=None):
    #     assert y is not None
    #     return model(x, t, y)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    mean_dice = 0.0
    mean_iou = 0.0

    for img in datal:
        # if number == 101:
        #     break
        if img[3] == 0:
            continue
        org_img = img[0]
        
        if vae == True:
            img[0] = encoder(img[0].float().to(dist_util.dev()))
            _max = img[0].max()
            _min = img[0].min()
            img[0] = (img[0] - _min) / (_max - _min) # [0, 1]
            img[0] = 2*img[0] - 1 # [-1, 1]

            print(f"min: {_min}")
            print(f"max: {_max}")

        print("Sample!!!")
        print(img[2].shape)
        mask = img[2].squeeze().numpy()  # squeeze() sẽ loại bỏ các chiều có size = 1
        # print(mask.shape)
        model_kwargs = {}
     #   img = next(data)  # should return an image from the dataloader "data"
        # print('img', img[0].shape, img[1])
        if args.dataset=='brats':
          Labelmask = th.where(img[2] > 0, 1, 0)
              
        else:
          print('img1', img[1])
          number=img[1]["path"]
          print('number', number)

        if args.class_cond:
            classes = th.randint(
                low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            print('y', model_kwargs["y"])
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        # print('samplefn', sample_fn)
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()
        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, 128, 128), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            # cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level
        )
        end.record()
        th.cuda.synchronize()
        th.cuda.current_stream().synchronize()

        # mask = img[2].squeeze().numpy()  # squeeze() sẽ loại bỏ các chiều có size = 1
        # # print(mask.shape)
        # model_kwargs = {}

        # classes = th.randint(
        #     low=0, high=1, size=(img[0].shape[0],), device=dist_util.dev()
        # )
        # model_kwargs["y"] = classes
        # # print('y', model_kwargs["y"])

        # sample_fn = (
        #     diffusion.ddim_sample_loop_known
        # )
        # # print('samplefn', sample_fn)
        # sample, x_noisy, org = sample_fn(
        #     model_fn,
        #     (img[0].shape[0], 4, 128, 128), img, org=img,
        #     clip_denoised=True,
        #     model_kwargs=model_kwargs,
        #     cond_fn=cond_fn,
        #     device=dist_util.dev(),
        #     noise_level=500
        # )
        if vae == True:
            sample = ((sample + 1) / 2) * (_max - _min) + _min
            sample = decoder(sample.float())
            sample = sample.detach()

        dice, iou = visualize_and_save(sample, org, number, mask)
        mean_dice += dice
        mean_iou += iou

        print("Dice Coefficient:", dice)
        print("IoU Score:", iou)
        # sample = sample.cpu().numpy()
        # org = org.cpu().numpy()
        # np.save(f'/data/hpc/minhdd/anomaly/src/sample/sample_{number}', sample)
        # np.save(f'/data/hpc/minhdd/anomaly/src/sample/org_{number}', org)
        # np.save(f'/data/hpc/minhdd/anomaly/src/sample/mask_{number}', mask)
        

        
        number += 1
        #   viz.image(visualize(sample[0,0, ...]), opts=dict(caption="sampled output0"))
        #   viz.image(visualize(sample[0,1, ...]), opts=dict(caption="sampled output1"))
        #   viz.image(visualize(sample[0,2, ...]), opts=dict(caption="sampled output2"))
        #   viz.image(visualize(sample[0,3, ...]), opts=dict(caption="sampled output3"))
        #   difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
        #   viz.heatmap(visualize(difftot), opts=dict(caption="difftot"))



        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    

    mean_dice = mean_dice / number
    mean_iou = mean_iou / number
    print("Mean Dice Coefficient:", mean_dice)
    print("Mean IoU Score:", mean_iou)
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="/data/hpc/minhdd/anomaly/data",
        clip_denoised=True,
        num_samples=25,
        batch_size=1,
        use_ddim=True,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        class_cond = True,
        dataset='brats',
        timestep_respacing = 'ddim1000'
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

