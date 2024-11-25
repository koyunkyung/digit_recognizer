from preprocess import get_data_loaders
from model import CNN
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(batch_size=64, num_epochs=10):
    # Load Kaggle DataLoaders
    train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size)

    # Define model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

if __name__ == "__main__":
    train_model()
