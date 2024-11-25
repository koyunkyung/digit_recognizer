### train the model and save the best weights ###

import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import get_data_loaders
from model import CNN
import os
import matplotlib.pyplot as plt

def train_model(epochs=10, learning_rate=0.001, batch_size=64, save_path="./outputs/best_model.pth"):
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size)

    # initialize model, loss, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # training loop
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / val_total)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc:{train_accuracies[-1]:.4f}")
        print(f"Val Loss: {val_losses[-1]:.4f}, Val Acc:{val_accuracies[-1]:.4f}")

    # save the best model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Modle saved to {save_path}")

    # plot training history
    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.savefig("./outputs/accuracy_plot.png")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("./outputs/loss_plot.png")

if __name__ == "__main__":
    train_model()