import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Mydataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
        dataset = []
        for i in range(len(labels)):
            temp_img = Image.open(images[i])
            temp_img = self.transform(temp_img)
            dataset.append((temp_img, labels[i]))
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.labels)


def load_dataset(self):
    data = []
    all_imgs_path = glob.glob(r'dataset\*.png')
    for ip in all_imgs_path:
        data.append(ip)
    labels = []
    for i in range(20):
        labels.extend([i] * 72)
    tr_imgs, te_imgs, tr_labs, te_labs = train_test_split(data, labels, train_size=0.9)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    tr_set = Mydataset(tr_imgs, tr_labs, transform)
    te_set = Mydataset(te_imgs, te_labs, transform)
    tr_loader = DataLoader(tr_set, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.workers,
                           pin_memory=True)
    te_loader = DataLoader(te_set, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.workers,
                           pin_memory=True)
    return tr_loader, te_loader
