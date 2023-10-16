import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import ImageFolder
import torch.multiprocessing
from torchvision import transforms as T
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def add_value(dict_obj, key, value):
    ''' Adds a key-value pair to the dictionary.
        If the key already exists in the dictionary, 
        it will associate multiple values with that 
        key instead of overwritting its value'''
    if key not in dict_obj:
        dict_obj[key] = value
    elif isinstance(dict_obj[key], list):
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [dict_obj[key], value]



class BubbleDataset(Dataset):
    def __init__(self, image_dir,transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.df = pd.read_csv('newdfrounded.csv',sep=',')


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        #!Taking only 1 channel of training image rather than 3
        #!Didnt work so changing back
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        #!Taking only 1 channel of training image rather than 3
        timestamp = Path(img_path).stem
        label=((self.df[self.df["timestamp"] == float(timestamp)].values[0][1] - min(self.df["Resistance"])) / (max(self.df["Resistance"] - min(self.df["Resistance"])) ))
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        return image, label



class SlugDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array([[0] [1] [0]])

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label
