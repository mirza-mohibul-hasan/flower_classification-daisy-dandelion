import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1000)
model.load_state_dict(torch.load('flower_classification_model.pth'))
model.eval()
# Create a new model with the correct final layer
new_model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
new_model.fc = nn.Linear(new_model.fc.in_features, 2)
# Copy the weights and biases from the loaded model to the new model
new_model.fc.weight.data = model.fc.weight.data[0:2]
new_model.fc.bias.data = model.fc.bias.data[0:2]

# Load and preprocess the unseen image
image_paths = ["test_daisy1.jpg", "test_daisy2.jpg","test_daisy3.jpg", "test_daisy4.jpg",  "test_dan1.jpg", "test_dan2.jpg", "test_dan3.jpg", "test_dan4.jpg",]
for image_path in image_paths:
    image = Image.open(image_path)
    preprocess = transforms.Compose(([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        output = model(input_batch)
    # Get the predicted class
    _,predicted_class = output.max(1)
    # Map the predicted class to the class name
    class_names = ['daisy', 'dandelion']
    predicted_class_name = class_names[predicted_class.item()]

    print(f'Prediceted class: {predicted_class_name}')
    # Display the image with the predicted class name
    image = np.array(image)
    plt.imshow(image)
    plt.axis('off')
    plt.text(10, 10,  f'Predicted: {predicted_class_name}', fontsize = 12, color="white", bbox=dict(facecolor='red', alpha=0.8))
    plt.show()