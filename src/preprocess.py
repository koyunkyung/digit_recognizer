import torch
from torchvision import transforms

def get_data_transforms():
    """ Returns data transformations for training and testing. """
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return train_transform, test_transform