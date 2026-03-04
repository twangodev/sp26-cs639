# CS 639 Assignment 2: Transformers

**Due 11:59PM CT on March 10th, 2026 on Gradescope**

In this assignment, you will build a small GPT-style language model from scratch, training it to speak Shakespearean. You will implement core transformer components like scaled dot-product attention, multi-head attention, and a full transformer block; train your model on a character-level text corpus; and run ablation experiments to understand why each component matters. You will also explore the effects of tokenization and positional encoding on language modeling.

## Google Colab Setup

The starter notebook is designed to run on Google Colab with a GPU.

- Colab defaults to a **CPU runtime**. You **must** switch to a GPU: go to `Runtime` → `Change runtime type` → `T4 GPU`. Training on CPU will be prohibitively slow.
- Colab may **disconnect** after periods of inactivity. If your runtime disconnects, your code is preserved, but all variables (including model weights) are lost. You will need to re-run all cells from the top to resume (each training is 5-10 minutes long).

## Estimated Run Time

All training is designed to run on Google Colab's free tier in under one hour:

1. Part A: ~11 min
2. Part B: 6 ablation runs × ~4 min each ≈ 24 min total
3. Part C: ~19 min (word-level model ~8 min + RoPE model ~11 min)

If your runtime is significantly longer, double-check that you are using GPU.

## Submission

You will submit to **two separate Gradescope assignments**:

1. **HW2 Writeup** (Gradescope): Upload a PDF containing your written answers, plots, and analysis for the questions marked with **[Writeup]** below.
2. **HW2 Code** (Gradescope): Upload your completed Colab notebook (`.ipynb`) with all code cells executed and outputs visible. Save your notebook after running all cells and make sure all outputs are visible. **Submissions with missing outputs will receive a 10-point deduction.**

Each subsection is labeled with its deliverable:

- **[Code]** — fill in the notebook; graded based on your code and outputs.
- **[Writeup]** — include your answer (plots, text, or both) in the PDF.
- **[Both]** — fill in the notebook code *and* include results/analysis in the PDF.

> **Note:** You are **not** allowed to change the pre-provided functions. Only the parts you are assigned to do.

## Model Configuration

The following model configuration is used throughout:

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 65 (character-level) |
| Embedding dimension (d_model) | 128 |
| Number of heads | 4 |
| Number of layers | 4 |
| Block size (context length) | 256 |
| Dropout | 0.1 |

## Part A: Core Transformer Implementation (60%)

In this part, you will implement the core building blocks of a GPT-style transformer. We provide the dataset, training loop, feed-forward network, and the outer GPT model shell. Your job is to implement the attention mechanism, the transformer block, and the GPT forward pass. The starter notebook contains detailed docstrings with expected input/output shapes for each function.

### A1: Scaled Dot-Product Attention (15 points) [Code]

Implement the function `scaled_dot_product_attention(Q, K, V, mask, dropout)`. Given query, key, and value matrices, your implementation should:

**(a)** Compute attention scores: QK^T / √d_k, where d_k is the dimension of each head.

**(b)** Apply the causal mask by setting masked positions to −∞ (before softmax).

**(c)** Apply softmax to obtain attention weights.

**(d)** Apply dropout to the attention weights.

**(e)** Return the weighted sum of value vectors.

### A2: Multi-Head Attention (15 points) [Code]

Implement the `MultiHeadAttention` module. This module should:

**(a)** Project the input into queries, keys, and values using three separate linear layers.

**(b)** Reshape into `n_heads` separate heads.

**(c)** Apply your `scaled_dot_product_attention` from A1 to each head.

**(d)** Concatenate the heads and apply a final output projection.

### A3: Transformer Block (15 points) [Code]

Implement the `TransformerBlock` module by wiring together:

**(a)** LayerNorm → Multi-Head Attention → Residual connection.

**(b)** LayerNorm → Feed-Forward Network → Residual connection.

Note that we use **Pre-LN** (LayerNorm before each sublayer), which is the GPT-2 convention. The `FeedForward` module is provided for you.

### A4: GPT Forward Pass (10 points) [Code]

Implement the `forward` method of the `GPT` class. We provide the `__init__` (which defines all layers) and the `generate` method. Your forward pass should:

**(a)** Look up token embeddings and positional embeddings, add them together, and apply dropout.

**(b)** Pass through the stack of transformer blocks.

**(c)** Apply the final LayerNorm and the language model head to produce logits.

**(d)** Compute the cross-entropy loss.

### A5: Training and Validation (5 points) [Both]

Run the provided training loop with your implementations for 5000 iterations. In your writeup, include:

**(a)** Your training and validation loss curves (the notebook will generate these).

**(b)** A sample of generated text from your trained model.

*Sanity check:* with a correct implementation, you should see a declining train and validation loss. **Before moving on to next parts, you need to make sure you pass this sanity check; as Part B and C depend on Part A.**

## Part B: Ablation Experiments (10%)

In this part, you will run ablation experiments to understand how key architectural choices affect your GPT model. **There is no code to write here** — the experiment code is provided in the starter notebook and you just need to run it. What we are testing is your ability to interpret results and understand the effect of key design choices.

To keep training time manageable, each ablation run uses **2000 iterations** (instead of the 5000 from Part A).

### B1: Varying Number of Attention Heads (5 points) [Writeup]

Train three models with different numbers of attention heads while keeping the embedding dimension fixed at d_model = 128:

- 1 head (d_k = 128)
- 4 heads (d_k = 32) — this is the default from Part A
- 8 heads (d_k = 16)

In your writeup:

**(a)** Include the combined validation loss plot (the notebook will generate this).

**(b)** Report the final validation loss for each configuration.

### B2: Varying Number of Layers (5 points) [Writeup]

Train three models with different depths while keeping all other hyperparameters fixed:

- 1 layer
- 2 layers
- 4 layers — this is the default from Part A

In your writeup:

**(a)** Include the combined validation loss plot.

**(b)** Report the final validation loss for each configuration.

**Writeup TODO.** Compare your results from B1 and B2. In 3–4 sentences, answer the following: Which architectural choice — number of heads or number of layers — had a bigger effect on performance? Why do you think one mattered more than the other at this model scale?

## Part C: Tokenization & Positional Encoding (30%)

In this part, you will explore two important design choices in transformer language models: tokenization and positional encoding.

### C1: Character-Level vs. Word-Level Tokenization (5 points) [Writeup]

In Part A, you trained a character-level model where each token is a single character (vocab size = 65). An alternative is **word-level tokenization**, where each token is a whitespace-separated word.

We provide a word-level tokenizer and a training run using the same model architecture in the starter notebook. Run both the character-level model (already trained in Part A) and the word-level model, then answer the following in your writeup:

**(a)** Fill in the following table comparing the two tokenization schemes:

|  | Character-level | Word-level |
|---|---|---|
| Vocabulary size | | |
| Avg. tokens per training example | | |
| Total model parameters | | |
| Final validation loss | | |

**(b)** Does comparing their validation losses directly make sense? Why or why not? *Hint:* what does a random baseline loss look like for each model? Recall that for uniform random predictions over a vocabulary of size V, the cross-entropy loss is ln(V).

### C2: Rotary Positional Embeddings (RoPE) (25 points) [Both]

In Part A, your model used **learned positional embeddings** — a separate embedding vector for each position. An alternative is **Rotary Positional Embeddings (RoPE)**, introduced by Su et al. (2021), which encodes position by rotating query and key vectors in attention.

The key idea: instead of adding position information to the input, RoPE applies a position-dependent rotation to each pair of dimensions in Q and K *inside* the attention computation. This means that the dot product q_i^T k_j naturally depends on the relative position i − j.

You need to implement the function that precomputes the rotation frequencies:

```
θ_d = 1 / base^(2d/d_k),  d = 0, 1, ..., d_k/2 - 1
```

For each position m and frequency θ_d, compute:

```
cos(m · θ_d)  and  sin(m · θ_d)
```

We provide the `GPT_RoPE` model variant. Your job is to implement:

1. `precompute_rope_frequencies`
2. `apply_rope`
3. `MultiHeadAttention_RoPE.forward`

The `apply_rope` function takes a tensor and the precomputed frequencies, splits each head's dimensions into pairs, and applies a 2D rotation to each pair using the corresponding cos and sin values.

After implementing, train the RoPE model and compare against the learned positional embedding model from Part A. In your writeup:

**(a)** Include the combined validation loss plot comparing learned PE vs. RoPE.

**(b)** Report final validation losses for both.

**(c)** In 2–3 sentences, compare the two approaches. What advantages might RoPE have over learned positional embeddings? (*Hint:* think about what happens when you encounter sequences longer than those seen during training.)