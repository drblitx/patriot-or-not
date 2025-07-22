# 🏈🏆 Patriot-Or-Not: Patriots Player Classifier, Dynasty Edition

Being a massive New England Patriots fan, I thought it'd be fun to create a simple deep learning app that classifies whether an image is of a New England Patriots player **during the Brady years (2000-2019) or not**. This project is built using PyTorch for model training and Streamlit for the interactive demo. 

## 🏟️ Project Goals
* Train a binary image classifier to distinguish Patriots players from other NFL players.
* Use transfer learning with a pretrained Convolutional Neural Network (CNN).
* Deploy a lightweight demo using Streamlit.

## 🔵 Example Use
Upload a photo of a football player and get a prediction like:
> Prediction: Patriots (92.3% confidence)

## ⚪️ Tech Stack
* Python
* PyTorch for training and inference
* Torchvision for datasets and pretrained models
* Pillow (PIL) for image loading
* Streamlit for frontend demo
* scikit-learn, NumPy, and matplotlib for evaluation

## 🔴 Folder Structure
```
project/
├── data/                  # not uploaded due to storage sizes; access photos via link in README
│   ├── other/
│   └── patriots/
│   └── sample_images/
├── model/
│   └── patriot_classifier.pth
├── app/
│   └── streamlit_app.py
├── train.py
├── predict.py
└── README.md
└── LICENSE
```

## 🐐 To-Do / Improvements
 * Add Grad-CAM visualizations
 * Improve accuracy with more images
 * Add image augmentation
 * Extend to multiclass (identify anything NFL/college that is not Patriots Brady-era players)