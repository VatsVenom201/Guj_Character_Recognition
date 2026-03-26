import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load dataset
    dataset = datasets.ImageFolder(
        root='preprocessed_images',
        transform=transform
    )

    # DataLoader (GPU optimized)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(len(train_loader))
    print('No. of classes',len(dataset.classes))

    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # 3. Freeze ALL layers first
    for param in model.parameters():
        param.requires_grad = False

    # 4. Unfreeze LAST 10 layers (your requirement)
    layers = list(model.features.children())

    for layer in layers[-10:]:
        for param in layer.parameters():
            param.requires_grad = True

    # 5. Replace Classifier (for 432 classes)

    num_classes = 432

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    #6. Move Model to GPU

    model = model.to(device)

    #7. Loss + Optimizer

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )


    from tqdm import tqdm

    EPOCHS = 10
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

if __name__ == "__main__":
    main()