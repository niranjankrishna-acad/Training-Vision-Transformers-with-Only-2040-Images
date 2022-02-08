from math import floor
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
labels_2_idx = {
    "Buttercup":0,
    "Colts Foot":1,
    "Daffodil":2,
    "Daisy":3,
    "Dandelion":4,
    "Fritilary":5,
    "Iris":6,
    "Pansy":7,
    "Sunflower":8,
    "Windflower":9,
    "Snowdrop":10,
    "Lily Valley":11,
    "Bluebell":12,
    "Crocus":13,
    "Tiger lily":14,
    "Tulip":15,
    "Cowslip":16
}
idx_2_labels = {}

for key, value in labels_2_idx.items():
    idx_2_labels[value] = key


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = open(annotations_file,"r").read()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = int(np.floor((idx + 1/17)))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        print(image, label)
        return image, label


dataset = CustomImageDataset(
    "data/files.txt",
    "data"
)
train_set, val_set = torch.utils.data.random_split(dataset, [1360 - 136, 136])


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=64, shuffle=True)