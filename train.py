"""
@name train.py
@author drblitx
@created July 2025
"""

# module imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split

# check if can use Apple's MPS, fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ImageNet mean and std (used by ResNet)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# transforms for train/validation sets
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),  # 224x224 for ResNet18
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# setting paths to data folders
data_dir = "data"
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms["train"])

# calcuate sizes & split randomly
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# apply different transforms to validation
val_dataset.dataset.transform = data_transforms["val"]

# create DataLoaders
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=8, shuffle=True),
    "val": DataLoader(val_dataset, batch_size=8, shuffle=False)
}

# dataset sizes/class names
dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
class_names = full_dataset.classes

# using ResNet18 for pretrained model with images
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# for param in model.parameters():
#     param.requires_grad=True

# replace final fully connected layer (og 1000 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) 

# move model to device/gpu
model = model.to(device)

# loss function; good w/ classifictation tasks with logits
criterion = nn.CrossEntropyLoss()

# optimize final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# training loop
num_epochs = 5

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 20)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero perimeter graidents
            optimizer.zero_grad()

            # forward pass
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward & optimize if in training phase only
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # stats tracker
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # epoch metrics
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.float() / dataset_sizes[phase]
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# saving the model
torch.save(model.state_dict(), "patriot_classifier.pth")
print("✅ Saved model to patriot_classifier.pth")
