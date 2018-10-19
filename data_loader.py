import math
import os

import torch
import torchvision
import torchvision.transforms.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def get_fashion_mnist(bs, size=32, train=True, mu=0.5, std=0.5):
    fatrans = transforms.Compose(
        [transforms.Grayscale(3),
         transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((mu, mu, mu), (std, std, std))]
    )
    fmset = torchvision.datasets.FashionMNIST("./data",
                                              train=train,
                                              transform=fatrans,
                                              download=True)
    fmloader = DataLoader(fmset,
                          batch_size=bs,
                          shuffle=True,
                          num_workers=2)
    return fmset, fmloader


def get_cifar10(bs, train=True, mu=0.5, std=0.5):
    citrans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((mu, mu, mu), (std, std, std))]
    )
    ciset = torchvision.datasets.CIFAR10("./data", train=train,
                                         transform=citrans,
                                         download=True)
    ciloader = DataLoader(ciset,
                          batch_size=bs,
                          shuffle=True,
                          num_workers=2)
    return ciset, ciloader


def get_classdata_cifar10(dataset, class_):
    classdata = torch.FloatTensor()
    for data in dataset:
        img, label = data
        if label == class_:
            classdata = torch.cat((classdata,
                                   torch.unsqueeze(img, 0)))
    return classdata


def get_celebA(bs, root_dir, size=32, test_split=0.2,
               train=True, shuffle=True, mu=0.5, std=0.5):
    catrans = transforms.Compose(
        [transforms.Resize(size),
         transforms.CenterCrop(size),
         transforms.ToTensor(),
         transforms.Normalize((mu, mu, mu), (std, std, std))]
    )

    caset = CelebADataset(root_dir, catrans,
                          test_split=test_split, train=train)
    caloader = DataLoader(caset,
                          batch_size=bs,
                          shuffle=shuffle,
                          num_workers=16)
    return caset, caloader


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform,
                 test_split=0.2, train=True):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.test_split = test_split
        self.train = train
        path, dir, files = os.walk(self.root_dir).__next__()
        self.size = math.floor(len(files) * 0.8) - 1 if train \
            else math.ceil(len(files) * 0.2) - 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx += 1
        im_name = "%06d.jpg" % (idx)
        path = os.path.join(self.root_dir, im_name)
        image = Image.open(path)
        return self.transform(image) if self.transform \
            else image
