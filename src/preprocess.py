### load and preprocess the MNIST dataset ###

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_transforms():
    """
    Defines transformations for training and testing datasets.
    - Normalizes pixel values to range [-1, 1]
    - Adds random augmentation (e.g. rotation) for training data

    Returns:
        train_transform: Transformations for training data
        test_transform: Transformations for test data
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),           # Augmentation: Rotate images randomly
        transforms.ToTensor(),                  # Convert PIL images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))    # Normalize to mean=0.5, std=0.5
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return train_transform, test_transform


def get_data_loaders(batch_size=64, data_dir="./data"):
    """
    Loads the MNIST dataset and returns DataLoader objects for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory to download and store the dataset.

    Returns:
        train_loader: Dataloader for training data.
        val_loader: Dataloader for validation data.
    """
    # get transformation
    train_transform, test_transform = get_data_transforms()

    # download and load datasets
    train_dataset = datasets.MNIST(root="./data", train=True, transform=train_transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=test_transform, download=True)

    # split training dataset into train/validation sets
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader