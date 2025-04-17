# Human Activity Recognition (HAR)

## Introduction
This project focuses on **Human Activity Recognition (HAR)** using machine learning techniques. HAR is an essential domain in artificial intelligence and wearable computing, with applications in healthcare, sports analytics, and smart environments. The goal is to classify human activities based on sensor data collected from wearable devices.

## Features
- **Data Processing**: Preprocessing and feature engineering techniques applied to raw sensor data.
- **Machine Learning Models**: Implementation and evaluation of multiple models for activity classification.
- **Performance Evaluation**: Model validation using metrics such as accuracy, precision, recall, and F1-score.
- **Deployment Readiness**: Structured code for potential real-world deployment.

## Dataset
The dataset consists of time-series sensor readings collected from accelerometers and gyroscopes. The data contains labeled activities such as walking, running, sitting, and standing.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn, TensorFlow/PyTorch (if deep learning is used)
- **Visualization**: Matplotlib, Seaborn
- **Notebook Environments**: Jupyter Notebook/Google Colab

## Implementation Steps
1. **Data Collection & Preprocessing**
   - Load and clean raw sensor data
   - Handle missing values and outliers
   - Normalize and standardize features
   
2. **Feature Engineering**
   - Extract statistical and time-domain features
   - Apply dimensionality reduction techniques (PCA, t-SNE)
   
3. **Model Training & Evaluation**
   - Train and test various machine learning models (e.g., Neural Networks (CNN and Bidirectional LSTM))
   - Evaluate models using cross-validation
   
4. **Results Analysis & Visualization**
   - Compare model performance metrics
   - Visualize results using confusion matrices and classification reports
   
5. **Future Enhancements**
   - Improve classification accuracy with advanced feature selection
   - Implement real-time HAR using embedded devices
   - Extend to multi-modal sensor data for improved robustness

## Contributors
- **[Hoàng Bảo Khanh]** - [github/hbkhanh22]
- **[Đinh Nguyễn Gia Bảo]** - [github/BAoD1nH]

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Build program
```bash
python <path-to-main.py> <type-of-dataset> <dataset-path>

---
Feel free to contribute to this project by submitting issues or pull requests!


