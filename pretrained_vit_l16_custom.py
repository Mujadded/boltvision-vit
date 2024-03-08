import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from pytorch_trainer.dataloaders import create_dataloaders
from pytorch_trainer.engine import train
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
import numpy as np
from collections import OrderedDict
import os

FileName = os.path.basename(__file__)

data_path = Path('data')
train_dir = data_path / 'train'
val_dir = data_path / 'val'
test_dir = data_path / 'test'
BATCH_SIZE = 16
EPOCHS = 1000


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

pretrained_vit_weights = torchvision.models.ViT_L_16_Weights.DEFAULT

pretrianed_vit = torchvision.models.vit_l_16(
    weights=pretrained_vit_weights).to(device)

vit_transforms = pretrained_vit_weights.transforms()


# Data augmentation we will do
train_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 90)),
    vit_transforms,
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    vit_transforms,
])

train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir, val_dir, test_dir, train_transforms, test_transforms, BATCH_SIZE)


for parameter in pretrianed_vit.parameters():
    parameter.requires_grad = False

# pretrianed_vit.heads = nn.Linear(
#     in_features=1024, out_features=len(class_names)).to(device)
pretrianed_vit.heads = nn.Sequential(OrderedDict([
    ('Linear1',  nn.Linear(in_features=1024, out_features=512)),
    ('Linear2',  nn.Linear(in_features=512, out_features=512)),
    ('Linear3',  nn.Linear(in_features=512, out_features=256)),
    ('Linear4',  nn.Linear(in_features=256, out_features=256)),
    ('Linear5',  nn.Linear(in_features=256, out_features=len(class_names))),
]))

optimizer = torch.optim.Adam(pretrianed_vit.parameters(),
                             lr=4e-4,
                             betas=(0.9, 0.999),
                             weight_decay=0.3)
loss_fn = torch.nn.CrossEntropyLoss()
# lr_lambda = lambda epoch : np.power(0.5, int(EPOCHS/25))
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
# scheduler = ReduceLROnPlateau(optimizer, 'min')
scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
train(
    pretrianed_vit,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer,
    loss_fn,
    EPOCHS,
    early_stopper_paitence=10,
    scheduler=scheduler
)
