import numpy as np


class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
        np.random.seed(0)
        self.W1 = np.random.uniform(-0.01, 0.01, (n_input, n_hidden))
        self.b1 = np.random.uniform(-0.01, 0.01, (1, n_hidden))
        self.W2 = np.random.uniform(-0.01, 0.01, (n_hidden, n_output))
        self.b2 = np.random.uniform(-0.01, 0.01, (1, n_output))

    def relu(self, Z):
        return np.maximum(0, Z)

    def forward(self, X):
        """O = ReLU(X @ W1 + b1) @ W2 + b2"""
        self.X = X
        self.Z1 = X @ self.W1 + self.b1 # Z1 = X*W1 + b1
        self.H = self.relu(self.Z1) # H = ReLU(Z1)
        self.O = self.H @ self.W2 + self.b2 # O = H*W2 + b2
        return self.O

    def softmax(self, O):
        """Subtract max first so it doesn't overflow."""
        max_logit = np.max(O, axis=1, keepdims=True)
        exp = np.exp(O - max_logit)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def cross_entropy_loss(self, probs, Y):
        """L = -mean(sum(Y * log(probs)))"""
        log_probs = np.log(probs + 1e-12) # add epsilon to avoid log(0)
        per_sample_loss = -np.sum(Y * log_probs, axis=1)
        return np.mean(per_sample_loss)

    def mse_loss(self, O, Y):
        """L = mean((O - Y)^2)"""
        diff = O - Y
        squared = diff ** 2
        return np.mean(squared)

    def backward(self, Y, lr, loss_type="ce"):
        """Compute gradients and update weights."""
        n = Y.shape[0]

        # gradient at the output
        if loss_type == "ce":
            probs = self.softmax(self.O)
            dO = (probs - Y) / n
        else:
            dO = 2 * (self.O - Y) / n

        # gradients for W2, b2
        dW2 = self.H.T @ dO
        db2 = np.sum(dO, axis=0, keepdims=True)

        # push gradient back through hidden layer
        dH = dO @ self.W2.T
        dZ1 = dH * (self.Z1 > 0) # zero out where ReLU was off

        # gradients for W1, b1
        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # update all weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def compute_loss(self, O, Y, loss_type):
        """Get cross-entropy or MSE loss."""
        if loss_type == "ce":
            probs = self.softmax(O)
            return self.cross_entropy_loss(probs, Y)
        else:
            return self.mse_loss(O, Y)

    def train_batch(self, X_batch, Y_batch, lr, loss_type):
        """Forward, loss, backward on one batch."""
        O = self.forward(X_batch)
        loss = self.compute_loss(O, Y_batch, loss_type)
        self.backward(Y_batch, lr, loss_type)
        return loss

    def shuffle(self, X, Y):
        """Shuffle X and Y together."""
        indices = np.random.permutation(X.shape[0])
        return X[indices], Y[indices]

    def make_batches(self, X, Y, batch_size):
        """Split data into mini-batches."""
        batches = []
        for start in range(0, X.shape[0], batch_size):
            X_batch = X[start:start + batch_size]
            Y_batch = Y[start:start + batch_size]
            batches.append((X_batch, Y_batch))
        return batches

    def train_epoch(self, X, Y, lr, batch_size, loss_type):
        """Shuffle, split into batches, train each one."""
        X_shuffled, Y_shuffled = self.shuffle(X, Y)
        batches = self.make_batches(X_shuffled, Y_shuffled, batch_size)

        for X_batch, Y_batch in batches:
            self.train_batch(X_batch, Y_batch, lr, loss_type)

    def train(self, X, Y, epochs, lr, batch_size=32, loss_type="ce"):
        """Train for n epochs, return list of losses."""
        np.random.seed(0)
        losses = []
        for epoch in range(epochs):
            self.train_epoch(X, Y, lr, batch_size, loss_type)
            # compute loss over all training data after each epoch
            O = self.forward(X)
            loss = self.compute_loss(O, Y, loss_type)
            losses.append(loss)
        return losses

    def evaluate(self, X, Y, loss_type="ce"):
        """Get loss (and accuracy for classification) without updating weights."""
        O = self.forward(X)

        if loss_type == "ce":
            probs = self.softmax(O)
            loss = self.cross_entropy_loss(probs, Y)
            predictions = np.argmax(O, axis=1)
            labels = np.argmax(Y, axis=1)
            accuracy = np.mean(predictions == labels)
            return loss, accuracy
        else:
            loss = self.mse_loss(O, Y)
            return loss