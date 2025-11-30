import glob
import re
import numpy as np
import os.path as osp
from torch.utils.data import Dataset


class LiTSDataset(Dataset):

    dataset_dir = "lits"
    dataset_url = "https://competitions.codalab.org/competitions/17094"

    
    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
        full_dataset: bool = False,
    ) -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.train_val_test_dir = train_val_test_dir
        self.full_dataset = full_dataset
        
        self.img_paths = []
        
        if train_val_test_dir:
            # Load data from specific split (train, val, or test)
            print(f"Loading LiTS data from split: {train_val_test_dir} - Full dataset: {full_dataset}")
            
            # Always load healthy data
            healthy_dir = osp.join(self.dataset_dir, "healthy", train_val_test_dir)
            if osp.exists(healthy_dir):
                healthy_paths = glob.glob(f"{healthy_dir}/*/healthy_vol_*_slice_*.npy")
                self.img_paths.extend(healthy_paths)
                print(f"Loaded {len(healthy_paths)} healthy images from {healthy_dir}")
            
            # Load unhealthy data if full_dataset is True
            if full_dataset:
                unhealthy_dir = osp.join(self.dataset_dir, "unhealthy", train_val_test_dir)
                if osp.exists(unhealthy_dir):
                    unhealthy_paths = glob.glob(f"{unhealthy_dir}/*/unhealthy_vol_*_slice_*.npy")
                    self.img_paths.extend(unhealthy_paths)
                    print(f"Loaded {len(unhealthy_paths)} unhealthy images from {unhealthy_dir}")
        else:
            # Load from all splits if no specific split is provided
            print(f"Loading LiTS data from all splits")
            splits = ['train', 'val', 'test']
            
            for split in splits:
                # Always load healthy data
                healthy_dir = osp.join(self.dataset_dir, "healthy", split)
                if osp.exists(healthy_dir):
                    healthy_paths = glob.glob(f"{healthy_dir}/*/healthy_vol_*_slice_*.npy")
                    self.img_paths.extend(healthy_paths)
                
                # Load unhealthy data if full_dataset is True
                if full_dataset:
                    unhealthy_dir = osp.join(self.dataset_dir, "unhealthy", split)
                    if osp.exists(unhealthy_dir):
                        unhealthy_paths = glob.glob(f"{unhealthy_dir}/*/unhealthy_vol_*_slice_*.npy")
                        self.img_paths.extend(unhealthy_paths)
        
        print(f"Total LiTS dataset size: {len(self.img_paths)} images")
        print(f"Full dataset mode: {full_dataset}")

    def prepare_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        
        # Load image
        image = np.load(image_path)

        if image.max() > 0:
            image = image / image.max()
        
        # Determine if this is healthy or unhealthy data
        # Check for 'unhealthy_vol' first because 'healthy_vol' is a substring of 'unhealthy_vol'
        is_unhealthy = 'unhealthy_vol' in osp.basename(image_path)
        is_healthy = not is_unhealthy and 'healthy_vol' in osp.basename(image_path)
        
        if is_healthy:
            # For healthy data, no mask exists, so create zero mask
            mask = np.zeros_like(image)
            mask = mask.astype(np.uint8)
            label = 0  # Healthy = 0
        else:
            # For unhealthy data, find corresponding mask
            # Convert unhealthy_vol_X_slice_Y.npy -> unhealthy_mask_X_slice_Y.npy
            mask_filename = osp.basename(image_path).replace('unhealthy_vol_', 'unhealthy_mask_')
            mask_dir = image_path.replace('/unhealthy/', '/masks/').replace(osp.basename(image_path), '')
            mask_path = osp.join(mask_dir, mask_filename)
            
            if osp.exists(mask_path):
                mask = np.load(mask_path)
                label = 1 if mask.max() > 0 else 0  # 1 if anomaly present, 0 if not
            else:
                print(f"Warning: Mask not found for {image_path}, expected at {mask_path}")
                mask = np.zeros_like(image)
                mask = mask.astype(np.uint8)

                label = 0
        
        out_dict = {"y": label}
        
        return image, out_dict, mask, label
        # return image, {'image': image} # For vae reconstruction


if __name__ == "__main__":
    # Test healthy only (train split)
    dataset_healthy = LiTSDataset(
        data_dir='/home/tqlong/qtung/gen-model-boilerplate/data/',
        train_val_test_dir='train',
        full_dataset=False
    )
    print(f"Healthy dataset size: {len(dataset_healthy)}")
    
    # Test full dataset (healthy + unhealthy) from train split
    dataset_full = LiTSDataset(
        data_dir='/home/tqlong/qtung/gen-model-boilerplate/data/',
        train_val_test_dir='train', 
        full_dataset=True
    )
    print(f"Full dataset size: {len(dataset_full)}")
    
    # Test a sample
    if len(dataset_healthy) > 0:
        image, cond, mask, label = dataset_healthy[0]
        print(f"Sample - Image shape: {image.shape}, Label: {label}, Mask max: {mask.max()}")
    
    if len(dataset_full) > 0 and len(dataset_full) > len(dataset_healthy):
        # Get an unhealthy sample
        unhealthy_idx = len(dataset_healthy)  # First unhealthy sample
        if unhealthy_idx < len(dataset_full):
            image, cond, mask, label = dataset_full[unhealthy_idx]
            print(f"Unhealthy sample - Image shape: {image.shape}, Label: {label}, Mask max: {mask.max()}")