import torch
import pandas as pd
from preprocess import get_data_loaders
from model import CNN

def generate_submission(model_path, output_file="./outputs/submission.csv", batch_size=64):
    """
    Generates a Kaggle submission file with predictions.

    Args:
        model_path (str): Path to the saved model file.
        output_file (str): Output CSV file for submission.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        None
    """
    # Get only the test_loader
    _, _, test_loader = get_data_loaders(batch_size=batch_size)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:  # Test loader does not have labels
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Create submission DataFrame
    submission = pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions})
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

if __name__ == "__main__":
    generate_submission(model_path="./outputs/best_model.pth")
