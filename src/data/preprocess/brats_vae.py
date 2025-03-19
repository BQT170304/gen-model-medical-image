import os
import os.path as osp
import random
import nibabel
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import glob
import numpy as np
import os.path as osp
from torch.utils.data import Dataset

random.seed(55)

data_dir = '/data/hpc/minhdd/anomaly/data'
dataset_dir = 'brats-2020'

# Define train, val, test directories
train_dir = osp.join(data_dir, dataset_dir, "train")
val_dir = osp.join(data_dir, dataset_dir, "val")
test_dir = osp.join(data_dir, dataset_dir, "test")

# Get img_paths from all directories
img_paths_train = glob.glob(f"{train_dir}/image/*/image_slice_*.npy")
img_paths_val = glob.glob(f"{val_dir}/image/*/image_slice_*.npy")
img_paths_test = glob.glob(f"{test_dir}/image/*/image_slice_*.npy")
print(len(img_paths_train) + len(img_paths_val) + len(img_paths_test))

unhealthy_paths = []
healthy_paths = [p for p in img_paths_train]

paths = img_paths_val + img_paths_test
for path in paths:
    mask_path = path.replace('image', 'mask')
    if os.path.exists(mask_path):
        mask = np.load(mask_path)

        if mask.max() > 0:
            unhealthy_paths.append(path)
        else:
            healthy_paths.append(path)
    else:
        healthy_paths.append(path)
            
print(f"healthy: {len(healthy_paths)}")
print(f"unhealthy: {len(unhealthy_paths)}")

# tvt_split: 0.7, 0.15, 0.15
random.shuffle(healthy_paths)
random.shuffle(unhealthy_paths)

# Handle unhealthy images first
train_unhealthy = unhealthy_paths[:800]
remaining_unhealthy = unhealthy_paths[800:]
# Split remaining unhealthy images evenly between val and test
val_unhealthy = remaining_unhealthy[:len(remaining_unhealthy)//2]
test_unhealthy = remaining_unhealthy[len(remaining_unhealthy)//2:]

total_size = len(healthy_paths) + len(unhealthy_paths)
train_healthy_count = int(0.7 * total_size) - 800
val_healthy_count = int(0.15 * total_size) - len(val_unhealthy)
test_healthy_count = len(healthy_paths) - train_healthy_count - val_healthy_count

train_healthy = healthy_paths[:train_healthy_count]
val_healthy = healthy_paths[train_healthy_count:train_healthy_count+val_healthy_count]
test_healthy = healthy_paths[train_healthy_count+val_healthy_count:]

train_paths = train_healthy + train_unhealthy
val_paths = val_healthy + val_unhealthy
test_paths = test_healthy + test_unhealthy

print("\nFinal split statistics:")
print(f"Train set: {len(train_paths)} images ({len(train_healthy)} healthy, {len(train_unhealthy)} unhealthy)")
print(f"Val set: {len(val_paths)} images ({len(val_healthy)} healthy, {len(val_unhealthy)} unhealthy)")
print(f"Test set: {len(test_paths)} images ({len(test_healthy)} healthy, {len(test_unhealthy)} unhealthy)")
print(f"Total: {len(train_paths) + len(val_paths) + len(test_paths)} images")

# Tao thu muc output
output_dir = '/data/hpc/qtung/gen-model-boilerplate/data/brats-2020'
train_output_dir = osp.join(output_dir, 'train')
val_output_dir = osp.join(output_dir, 'val')
test_output_dir = osp.join(output_dir, 'test')

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

def process_and_save_image(src_path, dest_dir):
    # Extract filename from source path
    filename = osp.basename(src_path)
    patient_id = osp.basename(osp.dirname(src_path))
    
    # Create patient directory in destination if it doesn't exist
    patient_dir = osp.join(dest_dir, 'image', patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Load and save image
    img = np.load(src_path)
    img_dest_path = osp.join(patient_dir, filename)
    np.save(img_dest_path, img)

print("\nProcessing and copying train set...")
for path in tqdm(train_paths):
    process_and_save_image(path, train_output_dir)

print("Processing and copying validation set...")
for path in tqdm(val_paths):
    process_and_save_image(path, val_output_dir)

print("Processing and copying test set...")
for path in tqdm(test_paths):
    process_and_save_image(path, test_output_dir)

print(f"Data processing complete. Files saved to {output_dir}")





