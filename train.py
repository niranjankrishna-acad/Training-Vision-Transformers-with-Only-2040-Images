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
import datetime
import time
import sys
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
        label = int(np.floor((counter)/80))
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

def accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    return (torch.sum(pred == target))


epochs = 512
os.makedirs("ckpt", exist_ok=True)

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
    counter = 0

    # training log
    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    acc = 0.0
    reader_start = time.time()
    batch_past = 0
    print_freq = 17

    aggregate_loss = 0
    for step, batch in enumerate(train_dataloader):

        train_reader_cost += time.time() - reader_start
        train_start = time.time()
        image, target =[x.cuda() for x in batch]
        image = image.to(torch.float)
        predictions = visionTransformer(image)
        x_a, x_b = randomAugmentation(image)
        z_embeddings_a = nn.functional.normalize(visionTransformer.module.vit(x_a), dim=1)
        z_embeddings_b =  nn.functional.normalize(visionTransformer.module.vit(x_b), dim=1)

        loss_1 = instanceDiscriminationLoss(predictions).mean()
        loss_2 = contrastiveLearningLoss(z_embeddings_a, z_embeddings_b).mean()

        total_loss = loss_1 + loss_2
        optimizer.zero_grad()
        total_loss.backward()
        aggregate_loss += total_loss.item()
        optimizer.step()

        train_run_cost += time.time() - train_start
        acc = accuracy(predictions, target).item()
        total_samples += image.shape[0]
        batch_past += 1

        if True:

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            msg = "[Epoch {}, iter: {}] acc: {:.5f}, lr: {:.5f}, loss: {:.5f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.".format(
                epoch, step, acc / batch_past,
                lr,
                total_loss.item(), train_reader_cost / batch_past,
                (train_reader_cost + train_run_cost) / batch_past,
                total_samples / batch_past,
                total_samples / (train_reader_cost + train_run_cost))
            print(msg)
            sys.stdout.flush()
            train_reader_cost = 0.0
            train_run_cost = 0.0
            total_samples = 0
            acc = 0.0
            batch_past = 0

        reader_start = time.time()

    torch.save(visionTransformer.state_dict(),"ckpt/model{}.pth".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("Total Epoch {} Loss : {}".format(epoch, aggregate_loss/len(train_dataloader)))
