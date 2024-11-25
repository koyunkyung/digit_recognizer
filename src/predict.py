import torch
import pandas as pd
from preprocess import get_data_loaders
from model import CNN

def generate_submission(model_path, output_file="submission.csv"):
    _, test_loader = get_data_loaders(batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    submission = pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions})
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

if __name__ == "__main__":
    generate_submission(model_path="./outputs/best_model.pth")
