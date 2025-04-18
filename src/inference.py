import torch
from lstm import MultiLayerBiLSTMClassifier
from feature_extraction import extract_features  # dùng để tạo đầu vào từ file
import numpy as np

def load_model(model_path="models/ucf11_lstm_model.pt"):
	model = MultiLayerBiLSTMClassifier(input_size=2048, hidden_size=256, num_layers=2, num_classes=11)
	model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
	model.eval()
	return model


# Dự đoán từ file input
def predict_activity(input_path, model):
	features = extract_features(input_path)  # chuyển CSV/video → vector
	input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # thêm batch dim
	with torch.no_grad():
		output = model(input_tensor)
		pred_class = torch.argmax(output, dim=1).item()
	return pred_class
