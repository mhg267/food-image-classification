import os
import torch
import cv2


from torch.utils.data import Dataset
from train_test_split import data_split
from PIL import Image

class FoodDataset(Dataset):
    def __init__(self, root, train="train", transform=None):
        self.root = root
        self.transform = transform

        train_data, val_data, test_data = data_split(root)

        if train == "train":
            self.dataset = train_data
        elif train == "val":
            self.dataset = val_data
        elif train == "test":
            self.dataset = test_data


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        image, label = self.dataset[index]

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label





