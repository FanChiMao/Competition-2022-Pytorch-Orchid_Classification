import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import csv
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

if __name__ == "__main__":

    # path
    csv_path = './dataset/label.csv'
    images_folder = './dataset/dataset_original/'
    save_folder = './dataset/new_dataset/'

    # CUBIC interpolation  cv::INTER_CUBIC = 2
    # augmented transform # cubic
    resize = A.Resize(480, 480, interpolation=2)
    no_change = A.HorizontalFlip(p=0)

    flip = A.HorizontalFlip(p=1)
    rot90 = A.Affine(rotate=90, p=1)
    rot180 = A.Affine(rotate=180, p=1)
    rot270 = A.Affine(rotate=270, p=1)
    fliprot90 = A.Compose([flip, rot90])
    fliprot180 = A.Compose([flip, rot180])
    fliprot270 = A.Compose([flip, rot270])

    transform_1 = A.Compose([resize])
    transform_2 = A.Compose([resize, flip])
    transform_3 = A.Compose([resize, rot90])
    transform_4 = A.Compose([resize, rot180])
    transform_5 = A.Compose([resize, rot270])
    transform_6 = A.Compose([resize, fliprot90])
    transform_7 = A.Compose([resize, fliprot180])
    transform_8 = A.Compose([resize, fliprot270])

    # transform_list x8
    transforms_list = [transform_1, transform_2, transform_3, transform_4,
                       transform_5, transform_6, transform_7, transform_8]

    df = pd.read_csv(csv_path)
    # len
    df_len = len(df)
    trans_len = len(transforms_list)

    # csv
    with open(f'{save_folder}/label.csv', 'w', encoding='UTF8', newline='') as f:

        # csv header
        fieldnames = ['filename', 'category']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_index in range(df_len):

            filename = df['filename'][image_index]
            label = df['category'][image_index]
            image_path = os.path.join(images_folder, filename)

            pillow_image = Image.open(image_path)
            image = np.array(pillow_image)

            for transform_index, transform in enumerate(transforms_list):

                    count = image_index * trans_len + transform_index + 1

                    # csv
                    row = [{'filename': f'{count}.jpg', 'category': label}]
                    writer.writerows(row)

                    # transform
                    image_aug = transform(image=image)['image']

                    # save image
                    img = Image.fromarray(image_aug, 'RGB')
                    img.save(f'{save_folder}/{count}.jpg')
