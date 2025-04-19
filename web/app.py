from flask import Flask, render_template, request, jsonify
import os
import sys
sys.path.append("../src")  # để import từ src

from inference import load_model, predict_activity

app = Flask(__name__)
model = load_model()

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	file = request.files['file']
	save_path = os.path.join("uploads", file.filename)
	file.save(save_path)

	pred = predict_activity(save_path, model)
	return jsonify({"prediction": str(pred)})

if __name__ == '__main__':
	app.run(debug=True)
