# HW1 Implementation Plan

## File Structure

```
hw1/
  hw1.py          # Main script — runs all experiments, produces all plots
  nn.py            # NeuralNetwork class
  data/            # Downloaded datasets
    iris.data
    housing.csv
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte
  plots/           # Output directory for saved figures
```

---

## Phase 1: Neural Network Class (`nn.py`)

### 1.1 — Constructor `__init__(self, n_input, n_hidden, n_output)`
- `np.random.seed(0)`
- W1: shape (n_input+1, n_hidden), uniform(-0.01, 0.01) — +1 for bias
- W2: shape (n_hidden+1, n_output), uniform(-0.01, 0.01) — +1 for bias

### 1.2 — Forward pass `forward(self, X)`
- Append bias column of 1s to X
- Z1 = X_bias @ W1
- H = ReLU(Z1)
- Append bias column of 1s to H
- O = H_bias @ W2 (linear output, no activation)
- Cache intermediates for backprop
- Return O (raw logits)

### 1.3 — Loss functions
- `softmax(O)` — numerically stable (subtract max before exp)
- `cross_entropy_loss(probs, Y_onehot)` — average over batch
- `mse_loss(O, Y)` — average over batch

### 1.4 — Backward pass `backward(self, Y, lr, loss_type)`
- Cross-entropy gradient: dO = (probs - Y_onehot) / batch_size
- MSE gradient: dO = 2*(O - Y) / batch_size
- dW2 = H_bias.T @ dO
- dH = (dO @ W2.T)[:, :-1]  (strip bias gradient)
- dZ1 = dH * (Z1 > 0)  (ReLU derivative)
- dW1 = X_bias.T @ dZ1
- W1 -= lr * dW1
- W2 -= lr * dW2

### 1.5 — Training loop `train(self, X, Y, epochs, lr, batch_size=32, loss_type="ce")`
- For each epoch: shuffle, iterate mini-batches, accumulate loss
- Return list of per-epoch average losses

### 1.6 — Evaluation `evaluate(self, X, Y, loss_type)`
- Forward pass only (no weight updates)
- Return average loss (and accuracy for classification)

---

## Phase 2: Data Loaders (`hw1.py`)

### 2.1 — Iris
- Load CSV with pandas (no header)
- Map species strings to one-hot (3 classes)
- `np.random.seed(0)`, shuffle, 80/20 split

### 2.2 — California Housing
- Load CSV with pandas
- Drop rows with NaN
- One-hot encode `ocean_proximity`
- Target = `median_house_value`, features = rest
- Standardize features (fit on train, transform both)
- `np.random.seed(0)`, shuffle, 80/20 split

### 2.3 — MNIST
- Use provided struct-based loader
- Flatten 28x28 -> 784, normalize /255
- Labels -> 10-element one-hot
- Use official 60k/10k split (or 20k subset if slow)

---

## Phase 3: Experiments (`hw1.py`)

### 3.1 — Q1: Iris experiments
- [ ] (a) LR comparison plot: LR in [1, 1e-2, 1e-3, 1e-8], hidden=5, 10 epochs
- [ ] (c) Test loss for each of the 4 LR models
- [ ] (d) Hidden size plot: sizes [2, 8, 16, 32], LR=1e-2, 10 epochs
- [ ] (e) Test loss + accuracy for each of the 4 hidden size models

### 3.2 — Q2: California Housing experiments
- [ ] Same (a)-(e) structure, but MSE loss, linear output
- [ ] Part (e): report MSE instead of accuracy

### 3.3 — Q3: MNIST experiments
- [ ] Same (a)-(e) structure, cross-entropy, 10 classes
- [ ] Use official train/test split

---

## Phase 4: Plotting helper

Each experiment needs a plot with:
- Title
- X-axis label (Epoch)
- Y-axis label (Average Loss)
- Legend (one line per hyperparameter value)
- Saved to `plots/` directory

---

## Build Order

1. `nn.py` — get the NN class working
2. Iris data loader + Q1 experiments — smallest dataset, fastest iteration
3. California Housing data loader + Q2 experiments
4. MNIST data loader + Q3 experiments
5. Final `hw1.py` that orchestrates everything

Written answers (Q1b, Q1c, Q1e, Q2b, Q2c, Q2e, Q3b, Q3c, Q3e, Q4, Q5) go in a separate PDF, not in code.