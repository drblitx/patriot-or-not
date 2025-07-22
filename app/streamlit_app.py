"""
@name streamlit_app.py
@author drblitx
@created July 2025
"""

# module imports
import streamlit as st
import torch
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np

# set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# class names
class_names = ['other', 'patriots']

# load model
@st.cache_resource
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    model.load_state_dict(torch.load("model/patriot_classifier.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# streamlit UI
st.markdown("## 🏈🏆 Patriots-Or-Not: Dynasty Edition")

uploaded_file = st.file_uploader("Upload a photo of an NFL player and find out if they're a **Brady-era Patriot**.", type=["jpg", "jpeg", "png"])

