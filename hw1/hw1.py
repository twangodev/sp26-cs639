import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import struct
from nn import NeuralNetwork

os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

EPOCHS = 10
LEARNING_RATES = [1, 1e-2, 1e-3, 1e-8]
HIDDEN_SIZES = [2, 8, 16, 32]


def load_iris():
    """Load Iris dataset, return X and Y (one-hot)."""
    df = pd.read_csv("data/iris.data", header=None)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values

    species = df["species"].values
    classes = sorted(set(species))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(species), len(classes)))
    for i, s in enumerate(species):
        Y[i, class_to_idx[s]] = 1

    return X, Y


def load_housing():
    """Load California Housing dataset, return X and Y."""
    df = pd.read_csv("data/housing.csv")
    df = df.dropna()

    # One-hot encode ocean_proximity
    dummies = pd.get_dummies(df["ocean_proximity"], dtype=float)
    df = pd.concat([df.drop("ocean_proximity", axis=1), dummies], axis=1)

    Y = df["median_house_value"].values.reshape(-1, 1)
    X = df.drop("median_house_value", axis=1).values

    return X, Y


def load_mnist():
    """Load MNIST dataset, return X_train, Y_train, X_test, Y_test."""
    def read_images(filename):
        with open(filename, 'rb') as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)

    def read_labels(filename):
        with open(filename, 'rb') as f:
            magic, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = read_images("data/train-images-idx3-ubyte").astype(float) / 255.0
    X_test = read_images("data/t10k-images-idx3-ubyte").astype(float) / 255.0
    y_train = read_labels("data/train-labels-idx1-ubyte")
    y_test = read_labels("data/t10k-labels-idx1-ubyte")

    Y_train = np.zeros((len(y_train), 10))
    for i, label in enumerate(y_train):
        Y_train[i, label] = 1

    Y_test = np.zeros((len(y_test), 10))
    for i, label in enumerate(y_test):
        Y_test[i, label] = 1

    return X_train, Y_train, X_test, Y_test


def split_data(X, Y, train_ratio=0.8):
    """Random 80/20 split with seed=0."""
    np.random.seed(0)
    indices = np.random.permutation(X.shape[0])
    split = int(X.shape[0] * train_ratio)
    return X[indices[:split]], Y[indices[:split]], X[indices[split:]], Y[indices[split:]]


def standardize(X_train, X_test):
    """Standardize to zero mean, unit variance. Fit on train only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def plot_losses(losses_dict, title, filename):
    """Plot per-epoch loss curves and save to plots/."""
    plt.figure()
    for label, losses in losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.savefig(f"plots/{filename}")
    plt.close()


def run_lr_experiment(X_train, Y_train, n_input, n_hidden, n_output, lrs, epochs, loss_type):
    """Train a model for each learning rate."""
    losses = {}
    models = {}
    for lr in lrs:
        nn = NeuralNetwork(n_input, n_hidden, n_output)
        losses[f"LR={lr}"] = nn.train(X_train, Y_train, epochs=epochs, lr=lr, loss_type=loss_type)
        models[lr] = nn
    return losses, models


def run_hidden_experiment(X_train, Y_train, n_input, n_output, hidden_sizes, lr, epochs, loss_type):
    """Train a model for each hidden size."""
    losses = {}
    models = {}
    for hs in hidden_sizes:
        nn = NeuralNetwork(n_input, hs, n_output)
        losses[f"Hidden={hs}"] = nn.train(X_train, Y_train, epochs=epochs, lr=lr, loss_type=loss_type)
        models[hs] = nn
    return losses, models


def print_eval(models, X_test, Y_test, loss_type, label_prefix):
    """Print test loss (and accuracy for classification) for each model."""
    for key, nn in models.items():
        result = nn.evaluate(X_test, Y_test, loss_type=loss_type)
        if loss_type == "ce":
            loss, acc = result
            print(f"  {label_prefix}={key}: loss={loss:.4f}, accuracy={acc:.4f}")
        else:
            print(f"  {label_prefix}={key}: MSE={result:.4f}")


def run_iris():
    X, Y = load_iris()
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    lr_losses, lr_models = run_lr_experiment(X_train, Y_train, 4, 5, 3, LEARNING_RATES, EPOCHS, "ce")
    plot_losses(lr_losses, "Iris: Training Loss by Learning Rate", "q1a_iris_lr.png")
    print("Q1(c) test losses:")
    print_eval(lr_models, X_test, Y_test, "ce", "LR")

    hs_losses, hs_models = run_hidden_experiment(X_train, Y_train, 4, 3, HIDDEN_SIZES, 1e-2, EPOCHS, "ce")
    plot_losses(hs_losses, "Iris: Training Loss by Hidden Size", "q1d_iris_hidden.png")
    print("Q1(e) test losses and accuracy:")
    print_eval(hs_models, X_test, Y_test, "ce", "Hidden")


def run_housing():
    X, Y = load_housing()
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    X_train, X_test = standardize(X_train, X_test)
    Y_train, Y_test = standardize(Y_train, Y_test)
    n_input = X_train.shape[1]

    lr_losses, lr_models = run_lr_experiment(X_train, Y_train, n_input, 5, 1, LEARNING_RATES, EPOCHS, "mse")
    plot_losses(lr_losses, "Housing: Training Loss by Learning Rate", "q2a_housing_lr.png")
    print("Q2(c) test losses:")
    print_eval(lr_models, X_test, Y_test, "mse", "LR")

    hs_losses, hs_models = run_hidden_experiment(X_train, Y_train, n_input, 1, HIDDEN_SIZES, 1e-2, EPOCHS, "mse")
    plot_losses(hs_losses, "Housing: Training Loss by Hidden Size", "q2d_housing_hidden.png")
    print("Q2(e) test MSE:")
    print_eval(hs_models, X_test, Y_test, "mse", "Hidden")


def run_mnist():
    X_train, Y_train, X_test, Y_test = load_mnist()

    lr_losses, lr_models = run_lr_experiment(X_train, Y_train, 784, 5, 10, LEARNING_RATES, EPOCHS, "ce")
    plot_losses(lr_losses, "MNIST: Training Loss by Learning Rate", "q3a_mnist_lr.png")
    print("Q3(c) test losses:")
    print_eval(lr_models, X_test, Y_test, "ce", "LR")

    hs_losses, hs_models = run_hidden_experiment(X_train, Y_train, 784, 10, HIDDEN_SIZES, 1e-2, EPOCHS, "ce")
    plot_losses(hs_losses, "MNIST: Training Loss by Hidden Size", "q3d_mnist_hidden.png")
    print("Q3(e) test losses and accuracy:")
    print_eval(hs_models, X_test, Y_test, "ce", "Hidden")


if __name__ == "__main__":
    run_iris()
    run_housing()
    run_mnist()