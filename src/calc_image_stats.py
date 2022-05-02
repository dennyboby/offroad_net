"""
ref:
1. https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
2. https://www.binarystudy.com/2021/04/how-to-calculate-mean-standard-deviation-images-pytorch.html
"""
# PACKAGES

import os
import numpy as np
import pandas as pd

import torchvision
from torchvision import transforms

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt


# PARAMS

# device = torch.device('cpu')
# num_workers = 4
# image_size = 512
# batch_size = 8
# data_path = '/kaggle/input/cassava-leaf-disease-classification/'
#
#
# class LeafData(Dataset):
#
#     def __init__(self,
#                  data,
#                  directory,
#                  transform=None):
#         self.data = data
#         self.directory = directory
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # import
#         path = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
#         image = cv2.imread(path, cv2.COLOR_BGR2RGB)
#
#         # augmentations
#         if self.transform is not None:
#             image = self.transform(image=image)['image']
#
#         return image
#
#
# # collapse-show
#
# augs = A.Compose([A.Resize(height=image_size,
#                            width=image_size),
#                   A.Normalize(mean=(0, 0, 0),
#                               std=(1, 1, 1)),
#                   ToTensorV2()])
#
# # EXAMINE SAMPLE BATCH
#
# # dataset
# image_dataset = LeafData(data=df,
#                          directory=data_path + 'train_images/',
#                          transform=augs)
#
# # data loader
# image_loader = DataLoader(image_dataset,
#                           batch_size=batch_size,
#                           shuffle=False,
#                           num_workers=num_workers,
#                           pin_memory=True)
#
# # display images
# for batch_idx, inputs in enumerate(image_loader):
#     fig = plt.figure(figsize=(14, 7))
#     for i in range(8):
#         ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
#         plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
#     break
#
# # COMPUTE MEAN / STD
#
# # placeholders
# psum = torch.tensor([0.0, 0.0, 0.0])
# psum_sq = torch.tensor([0.0, 0.0, 0.0])
#
# # loop through images
# for inputs in tqdm(image_loader):
#     psum += inputs.sum(axis=[0, 2, 3])
#     psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])
#
# # collapse-hide
#
# # Consider three vectors:
# # A = [1, 1]
# # B = [2, 2]
# # C = [1, 1, 2, 2]
#
# # Let's compute SDs in a classical way:
# # 1. Mean(A) = 1; Mean(B) = 2; Mean(C) = 1.5
# # 2. SD(A) = SD(B) = 0  # because there is no variation around the means
# # 3. SD(C) = sqrt(1/4 * ((1 - 1.5)**2 + (1 - 1.5)**2 + (1 - 1.5)**2 + (1 - 1.5)**2)) = 1/2
#
# # Note that SD(C) is clearly not equal to SD(A) + SD(B), which is zero.
#
# # Instead, we could compute SD(C) in three steps using the equation above:
# # 1. psum    = 1 + 1 + 2 + 2 = 6
# # 2. psum_sq = (1**2 + 1**2 + 2**2 + 2**2) = 10
# # 3. SD(C)   = sqrt((psum_sq - 1/N * psum**2) / N) = sqrt((10 - 36 / 4) / 4) = sqrt(1/4) = 1/2
#
# # We get the same result as in the classical way!
#
# # FINAL CALCULATIONS
#
# # pixel count
# count = len(df) * image_size * image_size
#
# # mean and std
# total_mean = psum / count
# total_var = (psum_sq / count) - (total_mean ** 2)
# total_std = torch.sqrt(total_var)
#
# # output
# print('mean: ' + str(total_mean))
# print('std:  ' + str(total_std))


def batch_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean, std


def get_image_data(data_root, img_dir="images"):
    data_path = os.path.join(data_root, img_dir)

    transform_img = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    image_data = torchvision.datasets.ImageFolder(
        root=data_path, transform=transform_img
    )

    return image_data


def calc_mean_std_dev(data_root, img_dir="images"):
    image_data = get_image_data(data_root, img_dir)

    image_data_loader = DataLoader(
        image_data,
        batch_size=len(image_data),
        shuffle=False,
        num_workers=0
    )

    return image_data_loader


def batch_mean_std(data_root, img_dir="images", batch_size=2):
    image_data = get_image_data(data_root, img_dir)

    loader = DataLoader(
        image_data,
        batch_size=batch_size,
        num_workers=1)

    mean, std = batch_mean_and_sd(loader)
    return mean, std


def mean_std(loader):
    images, labels = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    return mean, std


def main():
    list_datasets = [
        "yamaha_v1",
        "RUGD/RUGD_full",
    ]
    for data_root in list_datasets:
        image_data_loader = calc_mean_std_dev(data_root, img_dir="images")

        mean, std = mean_std(image_data_loader)
        print(f"Dataset: {data_root}"
              f"\nmean: \n{mean}"
              f"\nstd_dev: \n{std}")

        mean, std = mean_std(image_data_loader)
        scaled_mean = 255 * mean
        scaled_std_dev = 255 * std
        print(f"Dataset: {data_root}"
              f"\nscaled_mean: \n{scaled_mean}"
              f"\nscaled_std_dev: \n{scaled_std_dev}")


if __name__ == '__main__':
    main()
