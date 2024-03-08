import torch
import torchvision
from torchvision import transforms
from pytorch_trainer.dataloaders import create_dataloaders
from pytorch_trainer.engine import train
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from cct import cct_2

data_path = Path('data')
train_dir = data_path / 'train'
val_dir = data_path / 'val'
test_dir = data_path / 'test'
BATCH_SIZE = 8
EPOCHS = 1000


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


cct = cct_2(
    img_size = 224,
    n_conv_layers = 1,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_classes = 2,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)




# Data augmentation we will do
train_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 90)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),

])

train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(train_dir, val_dir, test_dir, train_transforms, test_transforms, BATCH_SIZE)




optimizer = torch.optim.Adam(cct.parameters(),
                            lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
# lr_lambda = lambda epoch : np.power(0.5, int(EPOCHS/25))
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scheduler = ReduceLROnPlateau(optimizer, 'min')
train(cct, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, EPOCHS, early_stopper_paitence=10, scheduler=scheduler)
