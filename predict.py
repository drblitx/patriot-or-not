"""
@name predict.py
@author drblitx
@created July 2025
"""

# module imports
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
from torchvision.models import resnet18, ResNet18_Weights

# set up device (same as in train.py)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# match class names
class_names = ['other', 'patriots']

# resetting up model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load("patriot_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# taking same training appraoch for model eval
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# image loading
img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # adding batch dimensions

# run the model
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    predicted_index = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")