import torch
import cv2
import numpy as np
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from lstm import MultiLayerBiLSTMClassifier
from preprocessing import preprocessingData
from feature_extraction import extract_features
import argparse
import os

def read_video_frames(video_path, num_frames=16):
	cap = cv2.VideoCapture(video_path)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# Chọn chỉ num_frames phân bố đều trong video
	frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
	frames = []
	for idx in range(total_frames):
		ret, frame = cap.read()
		if not ret:
			break
		if idx in frame_indices:
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frames.append(frame_rgb)
	cap.release()

	if len(frames) < num_frames:
		# Nếu không đủ frame, lặp lại frame cuối
		while len(frames) < num_frames:
			frames.append(frames[-1])

	return frames[:num_frames]

def load_model(model_path, input_size, hidden_size, num_layers, num_classes):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = MultiLayerBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

def inference(video_path, dataset, model_path):
	# Parameters (should match training)
	num_frames = 16
	hidden_size = 256
	num_layers = 2
	num_classes = 11 if dataset == "ucf11" else 50  # Sửa nếu dùng UCF101, HMDB51

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Step 1: Read video and extract frames
	frames = read_video_frames(video_path, num_frames)

	# Step 2: Transform frames
	transform = preprocessingData()
	transformed_frames = [transform(frame) for frame in frames]
	frames_tensor = torch.stack(transformed_frames, dim=0).to(device)

	# Step 3: Extract features using ResNet50
	resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
	resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-1])
	resnet.eval()
	with torch.no_grad():
		features_tensor = resnet_feat(frames_tensor)
	features = torch.flatten(features_tensor, start_dim=1).cpu().numpy()

	# Step 4: Load trained model
	input_size = features.shape[1]
	model = load_model(model_path, input_size, hidden_size, num_layers, num_classes)

	# Step 5: Inference
	with torch.no_grad():
		input_seq = torch.from_numpy(features).unsqueeze(0).float().to(device)  # (1, seq_len, feature_size)
		outputs = model(input_seq)
		predicted_class = torch.argmax(outputs, dim=1).item()

	print(f"Predicted class index: {predicted_class}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Inference on a single video using trained HAR model")
	parser.add_argument("dataset", type=str, help="Dataset used to train model (ucf11 or ucf50)")
	parser.add_argument("video_path", type=str, help="Path to input video file")
	parser.add_argument("model_path", type=str, help="Path to trained model (.pt)")
	args = parser.parse_args()

	inference(args.video_path, args.dataset.lower(), args.model_path)
