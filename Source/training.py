import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import os
from lstm import LSTMClassifier, MultiLayerBiLSTMClassifier

def loadFeatures(dataset):
    # Load the features and labels from numpy arrays
    train_features = torch.from_numpy(np.load(f'./features/{dataset}/train_features.npy')).float()
    train_labels = torch.from_numpy(np.load(f'./features/{dataset}/train_labels.npy'))
    idx = np.random.permutation(len(train_features))
    train_features, train_labels = train_features[idx], train_labels[idx]

    val_features = torch.from_numpy(np.load(f'./features/{dataset}/val_features.npy')).float()
    val_labels = torch.from_numpy(np.load(f'./features/{dataset}/val_labels.npy'))

    test_features = torch.from_numpy(np.load(f'./features/{dataset}/test_features.npy')).float()
    test_labels = torch.from_numpy(np.load(f'./features/{dataset}/test_labels.npy'))

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def trainModel(train_features, train_labels, val_features, val_labels, dataset, num_epochs=100, num_frames=32, hidden_size=256, learning_rate=0.0001):
    """
    Train the LSTM model with validation.
    """ 
    input_size = train_features.shape[-1]
    num_classes = len(np.unique(train_labels))

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    patience = 10  # Early stopping patience
    best_val_loss = float("inf")
    counter = 0

    # Instantiate the LSTM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_size, hidden_size, num_classes).cuda()
    model = MultiLayerBiLSTMClassifier(input_size, hidden_size, 2, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # Prepare DataLoaders
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=num_frames, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=num_frames, shuffle=False)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_total, train_correct = 0.0, 0, 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation step
        model.eval()
        total_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels.long())

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == batch_labels).sum().item()
                total_val += batch_labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset counter if validation loss improves
        else:
            counter += 1  # Increment counter if validation loss does not improve
        if counter >= patience:
            print("Early stopping triggered")
            break

        # Print training and validation results
        print(f'Epoch: [{epoch+1}/100] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds')

    # Save the training time in file .txt
    with open(f'./benchmarks/{dataset}/benchmark.txt', 'a') as f:
        f.write(f"Training time: {end_time - start_time:.2f} seconds\n")

    os.makedirs(f'./benchmarks/{dataset}', exist_ok=True)
    # Plot training and testing losses and accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./benchmarks/{dataset}/loss_plot.png')

    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'./benchmarks/{dataset}/accuracy_plot.png')
    plt.close()

    # 18/4/2025: Added to Save model weights
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), f'./models/{dataset}_lstm_model.pt')
    print(f'Model saved to ./models/{dataset}_lstm_model.pt')

    return model