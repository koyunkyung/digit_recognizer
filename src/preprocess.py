import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def normalize_tensor(tensor):
    """
    Normalize a tensor to the range [-1, 1].
    Args:
        tensor (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    return (tensor / 255.0 - 0.5) / 0.5  # Normalize to range [-1, 1]

def load_kaggle_data(train_path="./data/train.csv", test_path="./data/test.csv"):
    """
    Loads Kaggle's MNIST dataset from CSV files.

    Args:
        train_path (str): Path to the Kaggle train.csv file.
        test_path (str): Path to the Kaggle test.csv file.

    Returns:
        train_images: Tensor of training images.
        train_labels: Tensor of training labels.
        test_images: Tensor of test images.
    """
    # Load train.csv
    train_data = pd.read_csv(train_path)
    train_labels = torch.tensor(train_data["label"].values, dtype=torch.long)
    train_images = torch.tensor(train_data.drop("label", axis=1).values, dtype=torch.float32).reshape(-1, 1, 28, 28)

    # Load test.csv
    test_data = pd.read_csv(test_path)
    test_images = torch.tensor(test_data.values, dtype=torch.float32).reshape(-1, 1, 28, 28)

    return train_images, train_labels, test_images

def extract_from_subset(subset):
    """
    Extracts the tensors from a Subset object.
    Args:
        subset (torch.utils.data.Subset): Subset object containing the data.
    Returns:
        tensors: Tuple of extracted tensors (images, labels).
    """
    images = torch.stack([subset.dataset[i][0] for i in subset.indices])
    labels = torch.tensor([subset.dataset[i][1] for i in subset.indices])
    return images, labels

def get_data_loaders(batch_size=64, train_path="./data/train.csv", test_path="./data/test.csv"):
    """
    Creates DataLoader objects for training, validation, and testing datasets.

    Args:
        batch_size (int): Number of samples per batch.
        train_path (str): Path to Kaggle's train.csv file.
        test_path (str): Path to Kaggle's test.csv file.

    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
    """
    # Load Kaggle data
    train_images, train_labels, test_images = load_kaggle_data(train_path, test_path)

    # Normalize the images
    train_images = normalize_tensor(train_images)
    test_images = normalize_tensor(test_images)

    # Split training data into train/validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(train_images))
    val_size = len(train_images) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        list(zip(train_images, train_labels)), [train_size, val_size]
    )

    # Extract images and labels from subsets
    train_images, train_labels = extract_from_subset(train_subset)
    val_images, val_labels = extract_from_subset(val_subset)

    # Create DataLoader objects
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_images), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
