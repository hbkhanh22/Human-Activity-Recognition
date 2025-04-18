import torch
from lstm import YourLSTMModel  # đổi tên đúng model bạn dùng
from feature_extraction import extract_features  # dùng để tạo đầu vào từ file
import numpy as np

# Load model đã huấn luyện
def load_model(model_path="models/lstm_model.pt"):
	model = YourLSTMModel(...)
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
