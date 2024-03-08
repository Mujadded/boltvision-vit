import torch
import torchvision
from torchvision import transforms
from pytorch_trainer.dataloaders import create_dataloaders
from pytorch_trainer.engine import train
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

data_path = Path('data')
train_dir = data_path / 'train'
val_dir = data_path / 'val'
test_dir = data_path / 'test'
BATCH_SIZE = 16
EPOCHS = 1000


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

pretrianed_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

vit_transforms = pretrained_vit_weights.transforms()



# Data augmentation we will do
train_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 90)),
    vit_transforms,
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    vit_transforms,
])

train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(train_dir, val_dir, test_dir, train_transforms, test_transforms, BATCH_SIZE)



import torch.nn as nn
for parameter in pretrianed_vit.parameters():
    parameter.requires_grad = False

pretrianed_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

optimizer = torch.optim.Adam(pretrianed_vit.parameters(),
                            lr=1e-3,
                            betas=(0.9,0.999),
                            weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
# lr_lambda = lambda epoch : np.power(0.5, int(EPOCHS/25))
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scheduler = ReduceLROnPlateau(optimizer, 'min')
train(pretrianed_vit, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, EPOCHS, early_stopper_paitence=10, scheduler=scheduler)
