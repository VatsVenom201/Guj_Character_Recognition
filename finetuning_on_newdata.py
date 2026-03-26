import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():

    # ==============================
    # TRANSFORMS
    # ==============================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ==============================
    # LOAD NEW DATA
    # ==============================
    dataset = datasets.ImageFolder(
        root=r'C:\Users\vatsc\Projects\Practice\DeepLearning\Gujarati_Handwritten\dataset\preprocessed_images',
        transform=transform
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("Classes:", len(dataset.classes))
    print("Class mapping:")
    print(dataset.class_to_idx)

    # ==============================
    # MODEL SETUP
    # ==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(weights=None)

    # Replace classifier (same as before)
    num_classes = 432
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )

    # 🔥 LOAD YOUR TRAINED MODEL
    model.load_state_dict(torch.load(
        r"C:\Users\vatsc\Projects\Practice\DeepLearning\Gujarati_Handwritten\best_model.pth"
    ))

    model = model.to(device)

    # ==============================
    # FREEZE MOST LAYERS
    # ==============================
    for param in model.parameters():
        param.requires_grad = False

    # 🔥 Unfreeze LAST FEW layers (NOT all 10)
    layers = list(model.features.children())

    for layer in layers[-6:]:   # try 5–8 range
        for param in layer.parameters():
            param.requires_grad = True

    # Classifier always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    # ==============================
    # TRAINING SETUP
    # ==============================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4  # 🔥 LOWER LR for fine-tuning
    )

    # ==============================
    # TRAIN LOOP
    # ==============================
    EPOCHS = 10
    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0

        loop = tqdm(train_loader)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "finetuned_model.pth")
            print("✅ Best fine-tuned model saved!")

if __name__ == "__main__":
    main()