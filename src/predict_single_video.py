import os
import cv2
import numpy as np
import torch
from torchvision import models
import joblib
from preprocessing import preprocessingData

def extract_single_feature(video_path, transform, num_frames):
    # Load the pre-trained ResNet50 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)

    # Remove the last layer of the ResNet50 model to obtain the feature extractor
    resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        print(f"Less than {num_frames} frames !")
        return None

    frames_tensor = torch.stack(frames, dim=0).to(device)

    # Extract features
    with torch.no_grad():
        features_tensor = resnet_feat(frames_tensor)
    
    features = torch.flatten(features_tensor, start_dim=1).cpu().numpy()

    return np.mean(features, axis=0)


def predict_single_video(video_path, model_path, transform, num_frames):
    if not os.path.exists(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")
    
    # Load model
    model = joblib.load(model_path)

    # Preprocessing video and extracting features
    proccessed_data = preprocessingData()

    # Extract single feature
    video_features = extract_single_feature(video_path, transform, num_frames)

    # Predict the class of video


    
