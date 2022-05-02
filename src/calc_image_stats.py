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


# collapse-show

augs = A.Compose([A.Resize(height=image_size,
                           width=image_size),
                  A.Normalize(mean=(0, 0, 0),
                              std=(1, 1, 1)),
                  ToTensorV2()])

# EXAMINE SAMPLE BATCH

# dataset
image_dataset = LeafData(data=df,
                         directory=data_path + 'train_images/',
                         transform=augs)

# data loader
image_loader = DataLoader(image_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True)

# display images
for batch_idx, inputs in enumerate(image_loader):
    fig = plt.figure(figsize=(14, 7))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
        plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
    break

# COMPUTE MEAN / STD

# placeholders
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs in tqdm(image_loader):
    psum += inputs.sum(axis=[0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

# collapse-hide

# Consider three vectors:
# A = [1, 1]
# B = [2, 2]
# C = [1, 1, 2, 2]

# Let's compute SDs in a classical way:
# 1. Mean(A) = 1; Mean(B) = 2; Mean(C) = 1.5
# 2. SD(A) = SD(B) = 0  # because there is no variation around the means
# 3. SD(C) = sqrt(1/4 * ((1 - 1.5)**2 + (1 - 1.5)**2 + (1 - 1.5)**2 + (1 - 1.5)**2)) = 1/2

# Note that SD(C) is clearly not equal to SD(A) + SD(B), which is zero.

# Instead, we could compute SD(C) in three steps using the equation above:
# 1. psum    = 1 + 1 + 2 + 2 = 6
# 2. psum_sq = (1**2 + 1**2 + 2**2 + 2**2) = 10
# 3. SD(C)   = sqrt((psum_sq - 1/N * psum**2) / N) = sqrt((10 - 36 / 4) / 4) = sqrt(1/4) = 1/2

# We get the same result as in the classical way!

# FINAL CALCULATIONS

# pixel count
count = len(df) * image_size * image_size

# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

# output
print('mean: ' + str(total_mean))
print('std:  ' + str(total_std))
