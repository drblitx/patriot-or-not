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
class_names = ['Other', 'Patriots']

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
st.title("Patriots-Or-Not: Dynasty Edition")
st.image("banners.jpg", width=700)
st.markdown("""
        <div style="margin-bottom: 0px; font-size: 16px;">
            <b>Important:</b> This project is still a work in progress, as the training set is still pretty small (~600 images). 
            Some results may be incorrect. Banner image is from u/Nobiting's 
            <a href="https://www.reddit.com/r/Patriots/comments/arrdqe/updated_gillette_stadium_super_bowl_banners_6x/" target="_blank">
            post</a> on r/Patriots.
        </div>
        """, unsafe_allow_html=True)
st.markdown("""
<hr style="
    border: none;
    height: 4px;
    background: linear-gradient(to right, #C60C30, #A5ACAF, #002244);
    margin: 2em 0;
">
""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom: -1.5em; text-align: center; font-size: 20px;'> 
            Upload a photo of an NFL player to find out if they're a <b>Brady-era Patriot</b>.
            </div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", width=250)

    with col2:
        # preprocess the image
        input_tensor = transform(image).unsqueeze(0).to(device)

        # predict the image
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_index = torch.argmax(probabilities).item()
            predicted_class = class_names[predicted_index]
            confidence = probabilities[predicted_index].item() * 100

        st.markdown(f"""
        <div style='font-size: 1.5em; color: #002244; font-weight: bold; padding-left: 1em; line-height: 1.4;'>
            { "Brady-era Patriot" if predicted_class == "Patriots" else "Not a Brady-era Patriot" }<br>
            <span style='font-size: 1em; font-weight: normal; color: #444444;'>
                {confidence:.2f}% confidence
            </span>
        </div>
        """, unsafe_allow_html=True)

