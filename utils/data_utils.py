from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from .model_utils import transform_size

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


def dataset_indexes(num_classes, images_each_class, split=10, fold=1, shuffle=None):
    assert images_each_class % split == 0, f'not a perfect split'
    # folds means how many set of these folds will take use
    assert images_each_class % split == 0, f'not a perfect split'
    assert fold <= split, f'not a perfect split'

    output_list = []

    # total
    total_index_list = np.arange(num_classes * images_each_class)
    # print(len(train_index_list))
    test_index_list = []

    if shuffle:  # shuffle -->　np.random.shuffle
        test_indexes = np.array_split(np.random.shuffle(np.arange(images_each_class)), split)
    else:
        test_indexes = np.array_split(np.arange(images_each_class), split)

    for split_index in range(fold):
        test_index_list = []
        # of which fold
        test_index = test_indexes[split_index]

        for index in range(num_classes):
            test_index_list.extend(test_index + index * images_each_class)

        train_index_list = list(set(total_index_list) - set(test_index_list))

        # print(len(train_index_list))
        # print(len(test_index_list))
        output_list.append([train_index_list, test_index_list])

    return output_list

def set2loader(csv_path, images_folder, size: int, train_ids, test_ids, train_batch_size, test_batch_size):
    dataset = CustomDataset(csv_path, images_folder, transform=transform_size(size))
    train_subset = Subset(dataset, train_ids)
    test_subset = Subset(dataset, test_ids)
    trainloader = DataLoader(train_subset, batch_size=train_batch_size, pin_memory=True)
    testloader = DataLoader(test_subset, batch_size=test_batch_size, pin_memory=True)

    return trainloader, testloader