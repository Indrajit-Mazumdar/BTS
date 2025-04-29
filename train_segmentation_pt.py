import os
import sys
import math
import numpy as np
from glob import glob
import time
import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

from utils.configuration import config
from utils.losses_pt import *
from networks.ctcf_unet_3d import CTCFUNet3D
from networks.dbtc_net_3d import DBTCNet3D
from networks.mtc_net_3d import MTCNet3D


class BtsDataset(Dataset):

    def __init__(self, x_paths, y_paths):
        self.x_paths = x_paths
        self.y_paths = y_paths

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        image = np.load(self.x_paths[idx]).astype(np.float32)

        if config["network_dims"] == "2D":
            mask = np.load(self.y_paths[idx]).astype(np.float32)
        elif config["network_dims"] == "3D":
            mask = np.load(self.y_paths[idx]).astype(np.float32)

        return image, mask


if __name__ == '__main__':
    if config["gpu_cpu"] == "GPU" and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_dir = os.path.join(os.getcwd(), 'trained_models', str(config["training_year"]),
                             'Segmentation', config["seg_model_name"])
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    checkpoint_dir = os.path.join(model_dir, 'checkpoint')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if config["seg_model_name"] == "3D CTCF-UNet":
        model = CTCFUNet3D(in_channels=config["num_modalities"], out_channels=config["num_classes"])
    elif config["seg_model_name"] == "3D DBTC-Net":
        model = DBTCNet3D(in_channels=config["num_modalities"], out_channels=config["num_classes"])
    elif config["seg_model_name"] == "3D MTC-Net":
        model = MTCNet3D(in_channels=config["num_modalities"], out_channels=config["num_classes"])

    model = model.to(device)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["init_lr"], weight_decay=config["weight_decay"])

    loss_fn = soft_dice_loss

    x_train_paths = glob(config["train_path"] + "/images/*")
    y_train_paths = glob(config["train_path"] + "/masks/*")

    train_generator = BtsDataset(x_train_paths, y_train_paths)
    train_loader = DataLoader(dataset=train_generator,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              drop_last=True)

    init_epoch = 0
    num_epochs = config["num_epochs"]
    for epoch in range(init_epoch, num_epochs):
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{:04d}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
