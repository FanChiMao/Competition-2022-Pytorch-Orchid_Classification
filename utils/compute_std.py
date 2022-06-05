"""
For Calculating mean & standard of image dataset

dataset and transformation may vary, adjust for your own need.

Output : mean / std

Packages:
-torch
-tqdm
-pillow

Credit to:
https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# (cropped) image size
image_size = 320

# device
device = torch.device('cpu')

# path
csv_path = 'D:/pythonProject/Competition-2022-Pytorch-Orchid_Classification-main/dataset/aug_dataset/label.csv'
images_folder = 'D:/PycharmProjects/orchid_github/dataset/new_dataset/'

# data utils
class CustomDataset(Dataset):

    def __init__(self, csv_path, images_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df['filename'][index]
        label = self.df['category'][index]
        image_path = os.path.join(self.images_folder, filename)
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

transform =  transforms.Compose([
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])

dataset = CustomDataset(csv_path, images_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)


# placeholders
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs, labels in tqdm(dataloader):
    psum += inputs.sum(axis=[0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

# pixel count
count = len(dataloader) * image_size * image_size

# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

# output
print('mean: ' + str(total_mean))
print('std:  ' + str(total_std))