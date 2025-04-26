import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import os
from torchvision import models
import time
import json

def extract_features(samples, transform, dataset):
    print("Extracting features using ResNet50...")
    # Load the pre-trained ResNet50 model
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    resnet.eval()

    # Remove the last layer of the ResNet50 model to obtain the feature extractor
    resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet_feat.eval()

    processed_samples = []
    for frames, label in samples:
        transformed_frames = [transform(frame).to(device) for frame in frames]
        frames_tensor = torch.stack(transformed_frames, dim=0).to(device) 
        with torch.no_grad():
            features_tensor = resnet_feat(frames_tensor)  # Shape: (T, 2048, 1, 1)
            features = torch.flatten(features_tensor, start_dim=1).cpu().numpy()
        
        processed_samples.append((features, label))

    end_time = time.time()
    print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")

    os.makedirs(f'./benchmarks/{dataset}', exist_ok=True)

    # Save the feature extraction time in file .txt
    with open(f'./benchmarks/{dataset}/benchmark.txt', 'w') as f:
        f.write(f"Feature extraction time: {end_time - start_time:.2f} seconds\n")

    return processed_samples

def splittingData(samples, dataset):
    # Shuffle the samples
    np.random.shuffle(samples)

    # Split the samples into training and testing sets (80% training, 20% testing)
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    # Split the training sets into validation and training sets (80% training, 20% validation)
    validation_split_idx = int(0.8 * len(train_samples))
    train_samples, val_samples = train_samples[:validation_split_idx], train_samples[validation_split_idx:]

    # Separate features and labels for training, validation, and testing sets
    train_features, train_labels = zip(*train_samples)
    val_features, val_labels = zip(*val_samples)
    test_features, test_labels = zip(*test_samples)

    # Convert the labels to numerical labels using a LabelEncoder
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    val_labels = le.transform(val_labels)
    test_labels = le.transform(test_labels)

    train_features = np.array(train_features)
    val_features = np.array(val_features)
    test_features = np.array(test_features)

    # Print the shapes of the features and labels arrays
    print("Train Features shape:", train_features.shape)
    print("Train Labels shape:", train_labels.shape)
    print("Validation Features shape:", val_features.shape)
    print("Validation Labels shape:", val_labels.shape)
    print("Test Features shape:", test_features.shape)
    print("Test Labels shape:", test_labels.shape)

    os.makedirs(f'./features/{dataset}', exist_ok=True)

    # Save the features and labels to numpy arrays
    np.save(f'./features/{dataset}/train_features.npy', train_features)
    np.save(f'./features/{dataset}/train_labels.npy', train_labels)
    np.save(f'./features/{dataset}/val_features.npy', val_features)
    np.save(f'./features/{dataset}/val_labels.npy', val_labels)
    np.save(f'./features/{dataset}/test_features.npy', test_features)
    np.save(f'./features/{dataset}/test_labels.npy', test_labels)

    idx2label = {i: label for i, label in enumerate(le.classes_)}
    with open(f'./features/{dataset}/label_map_idx2label.json', 'w') as f:
        json.dump(idx2label, f, indent=4)

    # Save the LabelEncoder for later use
    return le
