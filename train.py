import enum
from math import floor, ceil
from pickletools import optimize
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from model import VisionTransformer, RandomAugmentation, InstanceDiscriminationLoss, ContrastiveLearningLoss
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
counter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = open(annotations_file,"r").read().split("\n")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        global counter
        counter += 1
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = read_image(img_path)
        label = int(np.floor((counter)/17))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


dataset = CustomImageDataset(
    "data/files.txt",
    "data",
    transform=transforms.Compose([transforms.Resize((224,224))])
)
train_set, val_set = torch.utils.data.random_split(dataset, [ceil(len(dataset) * 0.9), floor(len(dataset) * 0.1)])


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=64, shuffle=True)

epochs = 32

visionTransformer = nn.DataParallel(VisionTransformer(224, 10, 17))
visionTransformer.cuda()
randomAugmentation = nn.DataParallel(RandomAugmentation())
randomAugmentation .cuda()
instanceDiscriminationLoss = nn.DataParallel(InstanceDiscriminationLoss())
instanceDiscriminationLoss.cuda()
contrastiveLearningLoss = nn.DataParallel(ContrastiveLearningLoss())
contrastiveLearningLoss.cuda()
optimizer = torch.optim.AdamW(visionTransformer.parameters(), lr=0.001)
for epoch in range(epochs):
    aggregate_loss = 0
    for step, batch in enumerate(train_dataloader):
        
        batch = batch[0].to(torch.float)
        predictions = visionTransformer(batch)
        x_a, x_b = randomAugmentation(batch)
        z_embeddings_a = nn.functional.normalize(visionTransformer.module.vit(x_a), dim=1)
        z_embeddings_b =  nn.functional.normalize(visionTransformer.module.vit(x_b), dim=1)

        loss_1 = instanceDiscriminationLoss(predictions).mean()
        loss_2 = contrastiveLearningLoss(z_embeddings_a, z_embeddings_b).mean()

        total_loss = loss_1 + loss_2
        optimizer.zero_grad()
        total_loss.backward()
        aggregate_loss += total_loss.item()
        optimizer.step()
    torch.save(visionTransformer.state_dict(),"model")
    print("Total Epoch {} Loss : {}".format(epoch, aggregate_loss/len(train_dataloader)))
        