import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example model
model = nn.Linear(784, 10).to(device)

# Example dataset and DataLoader
train_loader = DataLoader(datasets.FakeData(transform=transforms.ToTensor()), batch_size=64)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        # Move data to the appropriate device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images.view(images.size(0), -1))  # Flatten images
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training complete!")
