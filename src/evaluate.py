import torch
from preprocess import get_data_loaders
from model import CNN

def evaluate_model(model_path, batch_size=64):
    """
    Evaluates the model on the test dataset.

    Args:
        model_path (str): Path to the saved model file (e.g., best_model.pth).
        batch_size (int): Batch size for the DataLoader.

    Returns:
        accuracy (float): Accuracy of the model on the test dataset.
    """
    # Get only the test_loader
    _, _, test_loader = get_data_loaders(batch_size=batch_size)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    evaluate_model(model_path="./outputs/best_model.pth")
