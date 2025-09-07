# **Diffusion-Models**

## **1. Key Features ✨**
This repository is a boilerplate for implementing and experimenting with various Generative Models, with a primary focus on Diffusion Models. The models are built from the ground up for research and experimentation purposes.
* Diffusion Model: Training with simple loss
* Inference with DDPM and  DDIM
* Using (label, image, text) as condition for diffusion model
* Latent diffusion: Image space to latent space with VAE
* Stable diffusion: Latent + Condition Diffusion
* Classifier-free guidance
* Medical Image Segmentation: using condition as medical image

## **2. Setup and Usage 🚀**

  ### **Clone the repository**
    https://github.com/BQT170304/gen-model-boilerplate
    
  ### **Install environment packages**
    cd gen-model-boilerplate
    conda create -n diffusion python=3.10
    conda activate diffusion 
    pip install -r requirements.txt

  ### **Training**
  set-up CUDA_VISIBLE_DEVICES and WANDB_API_KEY before training
  
    export CUDA_VISIBLE_DEVICES=0
    export WANDB_API_KEY=???

  choose from available experiments in folder "configs/experiment" or create your experiment to suit your task.
    
     # for generation task
    python src/train.py experiment=generation/latent_diffusion/train/brats2020_image trainer.devices=1

    # for reconstruction task
    python src/train.py experiment=reconstruction/vq_vae/train/brats2020_image trainer.devices=1

    # for segmentation task
    python src/train.py experiment=segmentation/condition_diffusion/eval/lidc trainer.devices=1

  ### **Evaluation**
  set-up CUDA_VISIBLE_DEVICES and WANDB_API_KEY before evaluating
  
    export CUDA_VISIBLE_DEVICES=0
    export WANDB_API_KEY=???
  
  choose from available experiments in folder "configs/experiment" or create your experiment to suit your task.
    
    # for generation task
    python src/eval.py experiment=generation/latent_diffusion/train/brats2020_image trainer.devices=1

    # for reconstruction task
    python src/eval.py experiment=reconstruction/vq_vae/train/brats2020_image trainer.devices=1

    # for segmentation task
    python src/eval.py experiment=segmentation/condition_diffusion/eval/lidc trainer.devices=1
  
  ### **Inference**
  Config dataset and model in inference code for specific usecases.
  
    python src/inference.py
    
## **3. Datasets 📊**

  - **Generation task**:
    - MNIST, FASHION-MNIST: 28x28 pixels
    - CIFAR10: 32x32 pixels
    - [GENDER](https://www.kaggle.com/datasets/yasserhessein/gender-dataset): 64x64 pixels
    - [CELEBA](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256): 256x256 pixels 
    - [AFHQ](https://www.kaggle.com/datasets/andrewmvd/animal-faces), [FFHQ](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set): 512x512 pixels
  
  - **Segmentation task**:
    - [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)
    - [BRATS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
    - [CVC-CLINIC](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
    - [ISIC](https://challenge.isic-archive.com/data/)

## **4. Architecture & Components 🛠️**
### **4.1. Attention**
  - Self Attention
  - Cross Attention
  - Spatial Transformer
  
### **4.2. Backbone**
  - ResNet Block
  - VGG Block
  - DenseNet Block
  - Inception Block

### **4.3 Embedder**
  - Time
  - Label: animal (dog, cat), number (0,1,...9), gender (male, female)
  - Image: Segmentation

### **4.4. Sampler**
  - DDPM: Denoising Diffusion Probabilistic Models
  - DDIM: Denoising Diffusion Implicit Models

### **4.5. Model**
  - Unet: Encoder, Decoder
  - Unconditional Diffusion Model
  - Conditional diffusion model (label, image, text - need to implement text embedder model)
  - Variational autoencoder: Vanilla (only work for reconstruction), VQVAE
  - Latent diffusion model
  - Latent conditional diffusion model with classifier-guidance (use label as condition)
