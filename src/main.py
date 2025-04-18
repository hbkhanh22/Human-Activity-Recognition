from load_dataset import readUCF, read_HMDB51
from preprocessing import preprocessingData
from feature_extraction import extract_features, splittingData
from training import loadFeatures, trainModel
from testing import testModel
from evaluation import modelEvaluation
import numpy as np 
import argparse


def main(dataset, data_dir):
    # Define the number of frames to extract features
    num_frames = 16

    if dataset.lower() == 'ucf11':
        # Load the UCF11 dataset
        samples = read_UCF11(data_dir, num_frames)
    elif dataset.lower() == 'ucf50' or dataset.lower() == 'ucf101':
        # Load the UCF50 dataset
        samples = read_UCF50(data_dir, num_frames)
    elif dataset.lower() == 'hmdb51':
        # Load the HMDB51 dataset
        samples = read_HMDB51(data_dir, num_frames)

    # Preprocess the data
    processed_data = preprocessingData()

    # Extract features using ResNet50
    processed_samples = extract_features(samples, processed_data, dataset.lower())

    # Split the data into training, validation, and testing sets
    le = splittingData(processed_samples, dataset.lower())

    # Load the features and labels for training, validation, and testing sets
    train_features, train_labels, val_features, val_labels, test_features, test_labels = loadFeatures(dataset.lower())

    # Train the model
    model = trainModel(train_features, train_labels, val_features, val_labels, dataset.lower())

    # Test the model
    y_pred, y_pred_proba = testModel(model, test_features, test_labels, dataset.lower(), num_frames)

    # Evaluate the model
    labels = np.arange(0, len(np.unique(test_labels)), 1)
    acc_score, pre_score, rec_score, f1, auc = modelEvaluation(y_pred, y_pred_proba, test_labels, labels, dataset.lower())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a model on UCF11 or UCF50 dataset.')
    parser.add_argument('dataset', type=str, help='Dataset to use (UCF11 or UCF50)')
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset')
    args = parser.parse_args()

    main(args.dataset, args.data_dir)
