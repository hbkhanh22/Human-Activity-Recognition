import torch
import cv2
import numpy as np
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from lstm import MultiLayerBiLSTMClassifier
from preprocessing import preprocessingData
import argparse
import os
import json

def load_label_map(dataset):
	label_path = f"src/label_map_idx2label_{dataset}.json"
	if not os.path.exists(label_path):
		raise FileNotFoundError(f"Label map not found: {label_path}")
	with open(label_path, "r", encoding="utf-8") as f:
		return json.load(f)

def read_video_frames(video_path, num_frames=16):
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video file: {video_path}")
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if total_frames == 0:
		raise RuntimeError(f"Video contains no frames: {video_path}")

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

	if len(frames) == 0:
		raise RuntimeError("No frames extracted from video.")
	while len(frames) < num_frames:
		frames.append(frames[-1])

	return frames[:num_frames]

def load_model(model_path, input_size, hidden_size, num_layers, num_classes):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = MultiLayerBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

def inference(dataset, video_path, model_path):
	num_frames = 32
	hidden_size = 256
	num_layers = 2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load label map and number of classes
	label_map = load_label_map(dataset)
	num_classes = len(label_map)

	# Step 1: Read and process video
	frames = read_video_frames(video_path, num_frames)
	transform = preprocessingData()
	transformed_frames = [transform(frame) for frame in frames]
	frames_tensor = torch.stack(transformed_frames, dim=0).to(device)

	# Step 2: Extract features
	resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
	resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-1])
	resnet.eval()
	with torch.no_grad():
		features_tensor = resnet_feat(frames_tensor)
	features = torch.flatten(features_tensor, start_dim=1).cpu().numpy()

	# Step 3: Load model
	input_size = features.shape[1]
	model = load_model(model_path, input_size, hidden_size, num_layers, num_classes)

	# Step 4: Predict
	with torch.no_grad():
		input_seq = torch.from_numpy(features).unsqueeze(0).float().to(device)
		outputs = model(input_seq)
		predicted_class = torch.argmax(outputs, dim=1).item()
		predicted_label = label_map[str(predicted_class)]

	print(f"Predicted class index: {predicted_class} ({predicted_label})")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Inference on a single video using trained HAR model")
	parser.add_argument("dataset", type=str, help="Dataset used to train model (ucf11 or ucf50)")
	parser.add_argument("video_path", type=str, help="Path to input video file")
	parser.add_argument("model_path", type=str, help="Path to trained model (.pt)")
	args = parser.parse_args()

	inference(args.dataset.lower(), args.video_path, args.model_path)
