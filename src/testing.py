import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def testModel(model, test_features, test_labels, num_frames=32):
    """
    "Test the LSTM model on the test set."
    """
    model.eval()
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=num_frames, shuffle=False)

    predicted_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y.long())
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())

            # Store softmax probabilities
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            all_outputs.append(softmax_outputs.cpu().numpy())

            test_correct += (predicted == batch_y).sum().item()
            test_total += batch_y.size(0)

    y_pred  = predicted_labels
    y_pred_proba = np.vstack(all_outputs)  # Convert list of arrays to a single numpy array

    test_loss /= len(test_loader)
    test_accuracy = 100 * test_correct / test_total

    # Print final testing results
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Save the test results to file .txt
    with open(f'./benchmarks/{dataset}/benchmark.txt', 'a') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

    return y_pred, y_pred_proba

