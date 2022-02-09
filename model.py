from ctypes import alignment
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from vit_pytorch import ViT


class VisionTransformer(nn.Module):
    def __init__(self, img_size, z_dim,num_classes):
        super(VisionTransformer, self).__init__()
        self.vit = ViT(
            image_size = img_size,
            patch_size = 32,
            num_classes = z_dim,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.linear = nn.Linear(in_features=z_dim, out_features=num_classes)
    def forward(self, x):
        x = self.vit(x)
        x = self.linear(x)
        x = nn.functional.softmax(x)
        return x

class RandomAugmentation(nn.Module):
    def __init__(self):
        super(RandomAugmentation, self).__init__()
        self.augment = transforms.Compose(
            [
                transforms.RandomRotation(35),
                transforms.ColorJitter(),

            ]
        )
    def forward(self, x):
        x_a = self.augment(x)
        x_b = self.augment(x)
        return x_a, x_b

class InstanceDiscriminationLoss(nn.Module):
    def __init__(self):
        super(InstanceDiscriminationLoss, self).__init__()

    def forward(self, predictions):
        return -torch.sum(torch.log(predictions))

    
class ContrastiveLearningLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLearningLoss, self).__init__()

    def forward(self, z_a, z_b):
        n = len(z_a)

        alignment = 0
        for i in range(n):
            alignment += -torch.sum(torch.dot(z_a[i].T, z_b[i]))
        uniformity_loss = 0
        for i in range(n):
            negative_sum = 0
            for j in range(n-1):
                if i == j:
                    continue
                negative_sum += torch.sum(torch.dot(z_a[i].T, z_b[i]))
            
            uniformity = torch.exp(torch.dot(z_a[i].T, z_b[i]))
            uniformity_loss += negative_sum + uniformity
        return alignment + uniformity_loss
