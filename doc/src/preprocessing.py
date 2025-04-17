import torchvision.transforms as transforms

def preprocessingData():
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Converts the frame from a NumPy array to a PIL Image, which is required for further transformations.
        transforms.Resize((224, 224)),  # Resizes the frame to 224x224 pixels, the input size expected by ResNet50.
        transforms.ToTensor(),  # Converts the PIL Image to a PyTorch tensor and scales pixel values to [0, 1].
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes the tensor using the mean and standard deviation of the ImageNet dataset, which ResNet50 was trained on.
    ])
    return transform