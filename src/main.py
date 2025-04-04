from load_dataset import readData
from preprocessing import preprocessingData
from feature_extraction import extract_features, splittingData
from training import loadFeatures, trainModel
from testing import testModel
from evaluation import modelEvaluation
import numpy as np 
import joblib
import os



def main():
    # Define the file path to your dataset
    # data_dir = './UCF11_updated_mpg/UCF11_updated_mpg'
    data_dir = './YouTube_Dataset_Annotated/action_youtube_naudio'
    # data_dir = './archive/UCF50'
    # Define the number of frames to extract features
    num_frames = 32

    # Define model path
    model_path = './src/model/model.pkl'

    # Load the dataset
    samples = readData(data_dir, num_frames)

    # Preprocess the data
    processed_data = preprocessingData()

    # Check if features already exist
    if os.path.exists('./src/features') and any(file.endswith('.npy') for file in os.listdir('./src/features')):
        print("Loading existing features...")
    else:
        # Extract features using ResNet50
        processed_samples = extract_features(samples, processed_data)

        # Split the data into training, validation, and testing sets
        le = splittingData(processed_samples)
        # Save the label encoder
        joblib.dump(le, './src/features/label_encoder.pkl')

    # Load the features and labels for training, validation, and testing sets
    train_features, train_labels, val_features, val_labels, test_features, test_labels = loadFeatures()


    # Check if the model already exists
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Load the label encoder
        le = joblib.load('./src/features/label_encoder.pkl')
        print("Loading existing model...")
        model = joblib.load(model_path)
    else:
        # Train the model
        model = trainModel(train_features, train_labels, val_features, val_labels)

        # Save the model
        print("Saving model...")
        joblib.dump(model, model_path)

    # Test the model
    y_pred, y_pred_proba = testModel(model, test_features, test_labels, num_frames)

    # Evaluate the model
    labels = le.inverse_transform(np.unique(test_labels))
    acc_score, pre_score, rec_score, f1, auc = modelEvaluation(y_pred, y_pred_proba, test_labels, labels)

main()