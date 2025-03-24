from typing import Any, Dict, Optional, Tuple

import torch
import pyrootutils
from torch import Tensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from albumentations import Compose

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dataset.brats2020 import BraTS2020Dataset


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Optional[Compose] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, cond, mask, label = self.dataset[idx]

        if self.transform is not None:
            # Apply the same transform to both image and mask
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, cond, mask, label


class BraTSDataModule(pl.LightningDataModule):
    """
    A DataModule for BraTS2020 dataset.
    
    Provides train, validation, and test dataloaders for the BraTS2020 dataset.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        train_val_test_dir: Tuple[str, str, str] = None,
        train_val_test_split: Tuple[int, int, int] = None,
        transform_train: Optional[Compose] = None,
        transform_val: Optional[Compose] = None,
        batch_size: int = 16,
        num_workers: int = 20,
        pin_memory: bool = False,
        dataset_name: str = 'brats2020',
        n_classes: str = 2,
        image_size: int = 128,
    ) -> None:
        """
        Args:
            data_dir: Root directory where the dataset is stored
            train_val_test_dir: Tuple of directory names for train, val and test
            train_val_test_split: Tuple with the number of samples to use for train, val, and test
            transform_train: Transforms to apply to the training data
            transform_val: Transforms to apply to validation and test data
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory during data loading
        """
        super().__init__()

        # Store all init parameters to hparams
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single process."""
        # BraTS dataset doesn't need to be downloaded here as it requires manual download
        pass
    
    def get_subset(self, dataset: Dataset, n_dataset: int):
        """Get a subset of the dataset for testing or limited training."""
        if 1 < n_dataset < len(dataset):
            print(f"Subsetting dataset from {len(dataset)} to {n_dataset} samples")
            return Subset(dataset, list(range(n_dataset)))
        
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and set up train, validation, and test datasets.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Adjust batch size for distributed training
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.hparams.batch_size
        
        # Debug prints
        print(f"Data dir: {self.hparams.data_dir}")
        
        # Load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.train_val_test_dir:
                train_dir, val_dir, test_dir = self.hparams.train_val_test_dir
                
                print(f"Train dir: {train_dir}")
                print(f"Val dir: {val_dir}")
                print(f"Test dir: {test_dir}")
                
                # Create datasets for each split
                train_set = BraTS2020Dataset(
                    data_dir=self.hparams.data_dir,
                    train_val_test_dir=train_dir
                )
                print(f"Train set size: {len(train_set)}")
                
                val_set = BraTS2020Dataset(
                    data_dir=self.hparams.data_dir,
                    train_val_test_dir=val_dir
                )
                print(f"Val set size: {len(val_set)}")
                
                test_set = BraTS2020Dataset(
                    data_dir=self.hparams.data_dir,
                    train_val_test_dir=test_dir
                )
                print(f"Test set size: {len(test_set)}")
                
                # Apply subset if specified
                if self.hparams.train_val_test_split is not None:
                    n_train, n_val, n_test = self.hparams.train_val_test_split
                    
                    train_set = self.get_subset(train_set, n_train)
                    val_set = self.get_subset(val_set, n_val)
                    test_set = self.get_subset(test_set, n_test)
            
            else:
                # If no split directories provided, use a single dataset and split it
                dataset = BraTS2020Dataset(data_dir=self.hparams.data_dir)
                
                if self.hparams.train_val_test_split:
                    # Use random split if proportions are provided
                    lengths = self.hparams.train_val_test_split
                    
                    # If lengths are provided as integers and sum up to less than the dataset size,
                    # use them as absolute counts
                    if isinstance(lengths[0], int) and sum(lengths) < len(dataset):
                        dataset = self.get_subset(dataset, sum(lengths))
                    
                    train_set, val_set, test_set = random_split(
                        dataset=dataset,
                        lengths=lengths,
                        generator=torch.Generator().manual_seed(42)
                    )
                else:
                    # Default split: 80% train, 10% val, 10% test
                    train_size = int(0.8 * len(dataset))
                    val_size = int(0.1 * len(dataset))
                    test_size = len(dataset) - train_size - val_size
                    
                    train_set, val_set, test_set = random_split(
                        dataset=dataset,
                        lengths=[train_size, val_size, test_size],
                        generator=torch.Generator().manual_seed(42)
                    )
            
            # Apply transforms to datasets
            self.data_train = TransformDataset(dataset=train_set, transform=self.hparams.transform_train)
            self.data_val = TransformDataset(dataset=val_set, transform=self.hparams.transform_val)
            self.data_test = TransformDataset(dataset=test_set, transform=self.hparams.transform_val)
            
            print(f'Train-Val-Test: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}')

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "data")

    @hydra.main(version_base=None, config_path=config_path, config_name="brats.yaml")
    def main(cfg: DictConfig):
        print(cfg)

        datamodule = hydra.utils.instantiate(cfg, data_dir=f"{root}/data")
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        print('train_dataloader:', len(train_dataloader))

        batch = next(iter(train_dataloader))
        image, cond, mask, label = batch
        
        print('Image shape:', image.shape, image.dtype)
        print('Mask shape:', mask.shape, mask.dtype)
        print('Label:', label)
        
        visualize(image, mask, label)

    def visualize(images, masks, labels):
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Take the first image from the batch for visualization
        image = images[0].numpy()
        mask = masks[0].numpy()
        label = labels[0].item()
        
        plt.figure(figsize=(16, 8))
        
        # Display the four MRI modalities
        plt.subplot(1, 5, 1)
        plt.imshow(image[..., 0], cmap='gray')
        plt.title('T1')
        
        plt.subplot(1, 5, 2)
        plt.imshow(image[..., 1], cmap='gray')
        plt.title('T1CE')
        
        plt.subplot(1, 5, 3)
        plt.imshow(image[..., 2], cmap='gray')
        plt.title('T2')
        
        plt.subplot(1, 5, 4)
        plt.imshow(image[..., 3], cmap='gray')
        plt.title('FLAIR')
        
        # Display the mask
        plt.subplot(1, 5, 5)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask (Label: {label})')
        
        plt.tight_layout()
        plt.show()

    main()
