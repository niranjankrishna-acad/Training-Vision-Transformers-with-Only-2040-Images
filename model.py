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
        x = x.flatten(x)
        x = self.linear(x)
        return x

class RandomAugmentation(nn.Module):
    def __init__(self):
        super(RandomAugmentation, self).__init__()
        self.augment = transforms.Compose(
            [
                transforms.RandomRotation(),
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

    
class ContrasiveLearningLoss(nn.Module):
    def __init__(self):
        super(ContrasiveLearningLoss, self).__init__()

    def forward(self, z_a, z_b):
        aligment = -torch.sum(torch.dot(z_a.T, z_b))
        uniformity_loss = 0
        n = len(z_a)
        for i in range(n):
            
            negative_sum = torch.sum(torch.dot(z_a[i].T.repeat(n - 1), z_b[:i] + z_b[i+1:]))
            uniformity = torch.exp(z_a[i].T, z_b[i])
            uniformity_loss += negative_sum + uniformity
        return aligment + uniformity_loss
