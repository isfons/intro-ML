import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class PeaksFunction:
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_noise(self, noise):
        self.y_train += self.rng.normal(0, noise, size=len(self.y_train))
        self.y_val += self.rng.normal(0, noise, size=len(self.y_val))

    def prepare_dataset(self, n_samples, test_size=0.5, batch_size=10, noise=0.1):
        X1, X2, Z = self.make_synthetic_data(n_samples)
        self.X_train, self.X_val, self.y_train, self.y_val = self.split_data(
            X1, X2, Z, test_size=test_size
        )

        # Agregar ruido
        self.add_noise(noise)

        # Normalizar inputs
        X_train_norm = self.scale_data(self.X_train)
        X_val_norm = self.scale_data(self.X_val)

        # Convertir a tensors de PyTorch
        X_train_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_norm).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val).reshape(-1, 1).to(self.device)

        # Data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    @staticmethod
    def peaks(X1, X2):
        return (
            -3
            * np.power((1 - X1), 2)
            * np.exp(-np.power(X1, 2) - np.power((X2 + 1), 2))
            - 10
            * (X1 / 5 - np.power(X1, 3) - np.power(X2, 5))
            * np.exp(-np.power(X1, 2) - np.power(X2, 2))
            - 1 / 3 * np.exp(-np.power((X1 + 1), 2) - np.power(X2, 2))
        )

    def make_synthetic_data(self, n_samples):
        sz = int(np.sqrt(n_samples))
        x = np.linspace(-3, 3, sz)
        y = np.linspace(-3, 3, sz)
        X1, X2 = np.meshgrid(x, y)
        Z = self.peaks(X1, X2)
        return X1, X2, Z

    def scale_data(self, X):
        return (X - (-3.0)) / (3.0 - (-3.0))

    def split_data(self, X1, X2, Z, test_size=0.5):
        # Flatten the meshgrid to create feature vectors
        X_flat = np.column_stack([X1.ravel(), X2.ravel()])
        y_flat = Z.ravel()

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y_flat, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def plot_scatter(self, color="red", label="Predictions"):
        X1, X2, Z = self.make_synthetic_data(n_samples=1000)

        fig = plt.figure(figsize=(12, 6), layout="constrained")
        ax = fig.add_subplot(121, projection="3d")
        ax.plot_surface(X1, X2, Z, edgecolor=None, color="lightgrey", zorder=1)
        ax.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            self.y_train,
            color=color,
            label=label,
            zorder=2,
        )
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("Z")
        plt.legend()
        return fig, ax

    def plot_predictions_surface(
        self, model, device, n_points=25, title="Model Predictions"
    ):
        """Plot model predictions as a surface"""
        # Create grid
        x = np.linspace(-3, 3, n_points)
        y = np.linspace(-3, 3, n_points)
        X1_grid, X2_grid = np.meshgrid(x, y)

        # Prepare input for model
        X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
        X_grid_norm = self.scale_data(X_grid)
        X_grid_tensor = torch.FloatTensor(X_grid_norm).to(device)

        # Get predictions
        model.eval()
        with torch.no_grad():
            Z_pred = model(X_grid_tensor).cpu().numpy().reshape(n_points, n_points)

        # Get true surface
        Z_true = self.peaks(X1_grid, X2_grid)

        # Plot
        fig = plt.figure(figsize=(16, 6), layout="constrained")

        # True surface
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot_surface(X1_grid, X2_grid, Z_true, cmap="viridis", alpha=0.8)
        ax1.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            self.y_train,
            color="red",
            s=50,
            label="Train Data",
        )
        ax1.set_title("True Surface + Training Data")
        ax1.set_xlabel("X1")
        ax1.set_ylabel("X2")
        ax1.set_zlabel("Z")
        ax1.legend()

        # Predicted surface
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.plot_surface(X1_grid, X2_grid, Z_pred, cmap="plasma", alpha=0.8)
        ax2.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            self.y_train,
            color="red",
            s=50,
            label="Train Data",
        )
        ax2.set_title("Model Predictions")
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        ax2.set_zlabel("Z")
        ax2.legend()

        fig.suptitle(title, fontsize=16, fontweight="bold")

        return fig


def plot_learning_curve(history, ax=None, title=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    ax.plot(history["train_loss"], label="Train Loss", color="b")
    ax.plot(history["val_loss"], label="Val Loss", color="r")
    ax.set_xlim(0, len(history["train_loss"]))
    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    if title:
        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
        )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_learning_curve_comparison(history_small, history_large):
    # Visualizar underfitting vs overfitting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Modelo pequeño (underfitting)
    plot_learning_curve(
        history_small,
        axes[0],
        title="Modelo Pequeño (UNDERFITTING)\nAmbas pérdidas altas y similares",
    )

    # Modelo grande (overfitting)
    plot_learning_curve(
        history_large,
        axes[1],
        title="Modelo Grande (OVERFITTING)\nVal loss diverge de train loss",
    )

    plt.tight_layout()
    plt.show()
