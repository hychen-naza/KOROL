import os
import numpy as np
import pandas as pd
import torch
import torchvision  
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle 
# This is the simplest possible residual block, with only one CNN layer.
# Looking at the paper, you can extend this block to have more layers, bottleneck, grouped convs (from shufflenet), etc.
# Or even look at more recent papers like resnext, regnet, resnest, senet, etc.
class DemoDataset(Dataset):
    def __init__(self, pickle_file_list):
        with open(pickle_file_list, 'rb') as file:
            self.demo_data = pickle.load(file)

    def __len__(self):
        return len(self.demo_data)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        return img, index

def get_dataloader(train_data_path, val_data_path, test_data_path):
    rgb_mean = (0.485, 0.456, 0.406)
    rgb_std = (0.229, 0.224, 0.225)
    transforms = torchvision.transforms.Compose([
      #torchvision.transforms.Resize((64,64)),
      #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
      torchvision.transforms.RandomHorizontalFlip(),
      #torchvision.transforms.RandomRotation(20, resample=Image.BILINEAR),
      torchvision.transforms.ToTensor(),
      #torchvision.transforms.Normalize(rgb_mean, rgb_std)
    ]) 

    train_dataset = DemoDataset(pickle_file_list)
    train_dataloader = DataLoader(train_dataset, batch_size=64, 
                                                   shuffle=True, num_workers=4)

    return train_dataloader

