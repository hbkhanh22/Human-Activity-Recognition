# src/inference_cli.py
import json
import torch
import argparse
import os
from lstm import MultiLayerBiLSTMClassifier
from feature_extraction import extract_features  # cần đảm bảo file này dùng được cho 1 video
from torchvision import transforms
from feature_extraction import extract_features_from_single_video

# --------------------- Load Model ---------------------
def load_model(model_path):
	model = MultiLayerBiLSTMClassifier(
		input_size=2048,
		hidden_size=256,
		num_layers=2,
		num_classes=11  # UCF11
	)
	model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
	model.eval()
	return model

# --------------------- Load Label Map ---------------------
def load_label_map(json_path="src/label_map_idx2label.json"):
	with open(json_path, "r") as f:
		label_map = json.load(f)
	return {int(k): v for k, v in label_map.items()}

# --------------------- Predict Activity ---------------------
def predict_activity(video_path, model):
	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225])
	])

	features = extract_features_from_single_video(video_path, transform)
	input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

	with torch.no_grad():
		output = model(input_tensor)
		pred = torch.argmax(output, dim=1).item()
	return pred

# --------------------- Main Entry ---------------------
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run HAR inference from a video path.")
	parser.add_argument("video_path", type=str, help="Path to video file for prediction")
	parser.add_argument("--model", type=str, default="src/models/ucf11_lstm_model.pt", help="Path to trained model")
	parser.add_argument("--label_map", type=str, default="src/label_map_idx2label.json", help="Path to label map JSON")
	args = parser.parse_args()

	if not os.path.isfile(args.video_path):
		raise FileNotFoundError(f"Video not found: {args.video_path}")
	if not os.path.isfile(args.model):
		raise FileNotFoundError(f"Model not found: {args.model}")
	if not os.path.isfile(args.label_map):
		raise FileNotFoundError(f"Label map not found: {args.label_map}")

	print(f"Loading model from {args.model}")
	model = load_model(args.model)

	print(f"Running inference on: {args.video_path}")
	result = predict_activity(args.video_path, model)

	label_map = load_label_map(args.label_map)
	print(f"Predicted activity class: {result}")
	print(f"Predicted activity: {label_map.get(result, 'Unknown')}")