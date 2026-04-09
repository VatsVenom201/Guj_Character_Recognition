# ============================================
# 1. IMPORTS
# ============================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import numpy as np

# ============================================
# 2. DUMMY DATASET (Replace with your real data)
# ============================================
X = torch.randn(2000, 20)
y = (X.sum(dim=1) > 0).long()

dataset = TensorDataset(X, y)

# ============================================
# 3. DYNAMIC MODEL BUILDER
# ============================================
class DynamicNN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size, dropout):
        super(DynamicNN, self).__init__()

        layers = []
        in_features = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, 2))  # output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ============================================
# 4. TRAINING FUNCTION (W&B CONTROLLED)
# ============================================
def train():
    wandb.init()

    config = wandb.config

    # DataLoader
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Model
    model = DynamicNN(
        input_dim=20,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        dropout=config.dropout
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer (dynamic)
    if config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

    # Training loop
    for epoch in range(config.epochs):
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in loader:
            optimizer.zero_grad()

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        acc = correct / total
        avg_loss = total_loss / len(loader)

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": acc
        })

# ============================================
# 5. SWEEP CONFIG (ALL 8+ PARAMS)
# ============================================
sweep_config = {
    "method": "random",

    "metric": {
        "name": "loss",
        "goal": "minimize"
    },

    "parameters": {

        # 🔥 Architecture
        "num_layers": {"values": [1, 2, 3, 4]},
        "hidden_size": {"values": [32, 64, 128, 256]},
        "dropout": {"values": [0.0, 0.2, 0.4, 0.5]},

        # 🔥 Optimization
        "optimizer": {"values": ["adam", "sgd"]},
        "lr": {"values": [1e-2, 1e-3, 1e-4]},
        "weight_decay": {"values": [0.0, 1e-4, 1e-3]},
        "momentum": {"values": [0.0, 0.9]},  # only for SGD

        # 🔥 Training
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"value": 5}
    }
}

# ============================================
# 6. RUN SWEEP
# ============================================
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="full-dynamic-pytorch")
    wandb.agent(sweep_id, train, count=10)