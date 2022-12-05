import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Mydataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        dataset = []
        for i in range(len(labels)):
            dataset.append((images[i], labels[i]))
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.labels)


def load_dataset():
    data =[]
    all_imgs_path=glob.glob(r'dataset\*\*.png')
    for var in all_imgs_path:
        print(var)

    labels = []
    for i in range(20):
        labels.extend([i] * 71)
    tr_imgs, te_imgs, tr_labs, te_labs = train_test_split(data, labels, train_size=0.9)
    tr_set = Mydataset(tr_imgs, tr_labs)
    te_set = Mydataset(te_imgs, te_labs)
    tr_loader =
    te_loader =
    return tr_loader, te_loader
