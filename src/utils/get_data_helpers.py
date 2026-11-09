from typing import Any

import pandas as pd
import numpy as np
import cv2
import os
import sys
import pathlib
from torch.utils.data import Dataset, DataLoader
import torch

from src.utils.processing_pipelines import training_transform

# assuming running from root

def get_data_path_from_dir(relative_path: pathlib.Path) -> pd.Series:
    """Load all pictures from a directory into a pd.Series"""
    return pd.Series([relative_path / file for file in os.listdir(relative_path)])

def get_raw_data():
    """
    returns raw data from data/raw/images in a tuple of tuple(train_df, test_df)
    """
    base_path = pathlib.Path("data/raw/images")
    raw_test_flip = get_data_path_from_dir(base_path / r"testing/flip")
    raw_test_notflip = get_data_path_from_dir(base_path / r"testing/notflip")
    raw_train_flip = get_data_path_from_dir(base_path / r"training/flip")
    raw_train_notflip = get_data_path_from_dir(base_path / r"training/notflip")
    train_df = pd.DataFrame({
        "images": pd.concat([raw_train_flip, raw_train_notflip], ignore_index=True),
        "labels": pd.concat([pd.Series(1, index=raw_train_flip.index), pd.Series(0, index=raw_train_notflip.index)], ignore_index=True)
        })
    test_df = pd.DataFrame({
        "images": pd.concat([raw_test_flip, raw_test_notflip], ignore_index=True),
        "labels": pd.concat([pd.Series(1, index=raw_test_flip.index), pd.Series(0, index=raw_test_notflip.index)], ignore_index=True)
        })

    return train_df, test_df


class LazyImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        will lazily store and load images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # lazy load when indexed
        img_path = self.image_paths[index]

        # Albumentations expects RGB/BGR NumPy arrays while cv2 reads as BGR
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        
        label = self.labels[index]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, torch.tensor(label, dtype=torch.long)



if __name__ == '__main__':
    # Add this line right below your imports
    torch.multiprocessing.set_sharing_strategy('file_system')

    # load pipeline
    train_df, test_df = get_raw_data()

    # making a dataset
    train_dataset = LazyImageDataset(image_paths=train_df['images'], labels=train_df['labels'], transform=training_transform)

    # making a data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    i = 0
    for images, labels in train_loader:
        print(i)
        i += 1
        print(images.size)
        print(labels.size)