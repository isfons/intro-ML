# Underfitting and Overfitting in Neural Networks
# Interactive notebook for chemical engineering undergraduates

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import ipywidgets as widgets
from IPython.display import display

# -------------------------------
# 1. Dataset: 1D nonlinear regression
# -------------------------------

def make_1d_data(n=200, noise=0.2, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n, 1)
    y = (
        torch.sin(2 * torch.pi * x)
        + 0.3 * torch.sin(6 * torch.pi * x)
        + noise * torch.randn_like(x)
    )
    return x, y

# -------------------------------
# 2. Train / validation split
# -------------------------------

def train_val_split(x, y, train_ratio=0.2):
    n_train = int(len(x) * train_ratio)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]

# -------------------------------
# 3. Model factory (slider-controlled)
# -------------------------------

class FlexibleNet(nn.Module):
    def __init__(self, hidden_units=10, depth=1):
        super().__init__()
        layers = []
        in_features = 1
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            in_features = hidden_units
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------
# 4. Training loop
# -------------------------------

def train_model(model, train_loader, val_loader, epochs=300, lr=1e-2):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

    return train_losses, val_losses, model

# -------------------------------
# 5. Visualization
# -------------------------------

def plot_results(x_train, y_train, x_val, y_val, model, train_losses, val_losses):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Learning curves
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Learning Curves")
    axes[0].legend()

    # Function fit
    x_plot = torch.linspace(0, 1, 300).unsqueeze(1)
    with torch.no_grad():
        y_plot = model(x_plot)

    axes[1].scatter(x_train, y_train, label="Train data", color="blue")
    axes[1].scatter(x_val, y_val, label="Validation data", color="orange")
    axes[1].plot(x_plot, y_plot, color="black", label="Model prediction")
    axes[1].set_title("Model Fit")
    axes[1].legend()

    plt.show()

# -------------------------------
# 6. Interactive widget
# -------------------------------

def run_experiment(hidden_units=5, depth=1, train_ratio=0.2, noise=0.2):
    x, y = make_1d_data(n=200, noise=noise)
    x_train, y_train, x_val, y_val = train_val_split(x, y, train_ratio)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))

    model = FlexibleNet(hidden_units=hidden_units, depth=depth)
    train_losses, val_losses, model = train_model(
        model, train_loader, val_loader
    )

    plot_results(x_train, y_train, x_val, y_val, model, train_losses, val_losses)

# -------------------------------
# 7. Sliders

widgets.interact(
    run_experiment,
    hidden_units=widgets.IntSlider(min=1, max=100, step=1, value=5, description="Hidden units"),
    depth=widgets.IntSlider(min=0, max=4, step=1, value=1, description="Depth"),
    train_ratio=widgets.FloatSlider(min=0.05, max=0.8, step=0.05, value=0.2, description="Train ratio"),
    noise=widgets.FloatSlider(min=0.0, max=0.6, step=0.05, value=0.2, description="Noise"),
)

# -------------------------------
# 8. Student questions and exercises
# -------------------------------

# Exercise 1: Underfitting
# -----------------------
# Set depth = 0 and hidden_units = 1.
# (a) What shape can this model represent?
# (b) Compare the training and validation losses. Why do they flatten early?
# (c) Would collecting more data help this model? Why or why not?

# Exercise 2: Increasing model capacity
# -----------------------
# Increase hidden_units gradually while keeping depth = 1.
# (a) At what point does the validation loss stop improving?
# (b) Why does increasing capacity initially help but eventually stop?

# Exercise 3: Overfitting by data scarcity
# -----------------------
# Set train_ratio = 0.05 and hidden_units >= 50.
# (a) Describe what happens to the training loss.
# (b) Describe what happens to the validation loss.
# (c) How does the model prediction look between training points?

# Exercise 4: Effect of noise
# -----------------------
# Fix the model (e.g., hidden_units = 50, depth = 3) and vary the noise level.
# (a) How does increasing noise affect the best achievable validation loss?
# (b) Does the model still try to fit the noise?

# Exercise 5: Bias–variance tradeoff (conceptual)
# -----------------------
# For each of the following cases, classify the model as high-bias or high-variance:
# (a) Linear model on this dataset
# (b) Deep network with very little data
# (c) Moderate-sized network with sufficient data

# Bonus question (chemical engineering intuition)
# -----------------------
# This dataset represents an unknown nonlinear process.
# (a) What does overfitting correspond to in terms of process modeling?
# (b) Why is validating on unseen operating conditions critical in practice?
