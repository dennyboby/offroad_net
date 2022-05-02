# collapse-hide
# ref: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
# PACKAGES

import os
import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt

# PARAMS

device = torch.device('cpu')
num_workers = 4
image_size = 512
batch_size = 8
data_path = '/kaggle/input/cassava-leaf-disease-classification/'


class LeafData(Dataset):

    def __init__(self,
                 data,
                 directory,
                 transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # import
        path = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image
