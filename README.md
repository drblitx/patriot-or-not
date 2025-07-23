# 🏈🏆 Patriot-Or-Not: Patriots Classifier, Dynasty Edition

Being a massive New England Patriots fan, I thought it'd be fun to create a simple deep learning app that classifies whether an image is of a New England Patriots player **during the Brady years (2000-2019) or not**. This project is built using PyTorch for model training and Streamlit for the interactive demo. 

## 🔗 https://patriot-or-not.streamlit.app

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

## 🔴 Folder Structure
```
project/
├── app/
│   └── streamlit_app.py
├── model/
│   └── patriot_classifier.pth
├── data/                         # not uploaded due to storage sizes; access via link
│   ├── other/
│   ├── patriots/
│   └── sample_images/
├── LICENSE
├── README.md
├── banners.jpg
├── photo_cleaner.py
├── predict.py
├── requirements.txt
└── train.py
```

## 🎞️ Accessing Images Used
To see the images that were used in this project, use this link: [patriots-or-not images](https://drive.google.com/drive/folders/1-HEccfBjIeWEabulW216z7I7QC4WsYua?usp=sharing)

**Image Information**
* `patriots` has 326 images in 224x224 format
* `other` has 326 images in 224x224 format
* `sample_images` (for Streamlit) has 11 images in original formats

## 🐐 To-Do / Improvements
 * scikit-learn, NumPy, and matplotlib for evaluation
 * Add Grad-CAM visualizations
 * Improve accuracy with more images
 * Add image augmentation
 * Extend to multiclass (identify anything NFL/college that is not Patriots Brady-era players)
