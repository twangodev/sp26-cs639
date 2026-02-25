# CS639 HW1 Written Answers

## Iris

1. Training Loss by LR
	 ![[q1a_iris_lr.png]]
2.  LR=1 is the only one with changes in training loss, but it oscillates back and forth. The other three LRs show almost no change in 10 epochs probably just because they are too small.
3. Datapoints:
	- LR=1: 1.0966, accuracy=33.3%
	- LR=0.01: 1.0987, accuracy=30.0%
	- LR=0.001: 1.0987, accuracy=30.0%
	- LR=1e-8: 1.0987, accuracy=30.0%

	LR=1 has the lowest test loss, matching the training plot. 33% accuracy is basically random for 3 classes.

4. Training Loss by Hidden Size
	![[q1d_iris_hidden.png]]
5. Average Test Loss:
	- Hidden=2: loss=1.0988, accuracy=30.0%
	- Hidden=8: loss=1.0986, accuracy=30.0%
	- Hidden=16: loss=1.0987, accuracy=30.0%
	- Hidden=32: loss=1.0985, accuracy=30.0%

	They are all about the same. The training plot shows slightly faster decrease for larger size though, but it probably is'nt enough data since it has trained only over 10 epochs.

## California Housing

1. Training Loss by LR
	 ![[q2a_housing_lr.png]]
2. LR=0.01 is the best. It drops quickly from ~0.45 to ~0.31. LR=0.001 is more slow. LR=1 diverges (NaN). LR=1e-8 stays flat.
3. Datapoints:
	- LR=1: NaN (diverged)
	- LR=0.01: MSE=0.3262
	- LR=0.001: MSE=0.4454
	- LR=1e-8: MSE=0.9793

	LR=0.01 is best. LR=0.001 still converging. Test losses track training losses, no overfitting.

4. Training Loss by Hidden Size
	![[q2d_housing_hidden.png]]
5. Average Test MSE:
	- Hidden=2: MSE=0.3642
	- Hidden=8: MSE=0.3037
	- Hidden=16: MSE=0.2957
	- Hidden=32: MSE=0.2928

	Bigger hidden layers do better. Hidden=2 underfits — not enough capacity. Hidden=8/16/32 are similar with diminishing returns. No overfitting since the dataset is large relative to model size.

## MNIST

1. Training Loss by LR
	 ![[q3a_mnist_lr.png]]
2. LR=0.01 is the best — smooth drop from ~1.0 to ~0.4. LR=0.001 learns but slowly, still at ~0.95 by epoch 10. LR=1 oscillates around 1.2-1.4. LR=1e-8 is flat at 2.3 (-ln(1/10), random baseline).
3. Datapoints:
	- LR=1: loss=1.2170, accuracy=49.8%
	- LR=0.01: loss=0.3831, accuracy=88.8%
	- LR=0.001: loss=0.9656, accuracy=71.7%
	- LR=1e-8: loss=2.3025, accuracy=9.8%

	LR=0.01 wins with 88.8%. LR=0.001 is still learning (71.7%). LR=1 is unstable (49.8%). LR=1e-8 is random (9.8%). Test results match training losses.

4. Training Loss by Hidden Size
	![[q3d_mnist_hidden.png]]
5. Average Test Loss and Accuracy:
	- Hidden=2: loss=1.0502, accuracy=63.4%
	- Hidden=8: loss=0.3172, accuracy=91.2%
	- Hidden=16: loss=0.2729, accuracy=92.2%
	- Hidden=32: loss=0.2305, accuracy=93.4%

	More hidden units = better. Hidden=2 is a bottleneck — 784 dims compressed to 2 loses too much info. Big jump from 2 to 8, then diminishing returns. Training curves show the same ordering.

## Comparison

Housing was definitely the easiest, MNIST was okay, Iris was the hardest. Iris barely learned in 10 epochs due to only 4 features with tiny init weights. MNIST learned well thanks to more data (60k samples) and higher dimensionality (784 features). We can improve this by just increasing epochs I believe.

## Debugging

1. **Training loss computed per-batch instead of per-epoch**
	Was averaging batch losses during training, but each batch sees the model at a different point. Changed to computing loss over the full training set after each epoch.

2. **MSE overflow**
	```
	Q2(c): LR=0.01: MSE=nan
	```
	Target values are ~15k-500k, so (0 - 400000)^2 overflows. Fixed by standardizing Y in addition to X.

3. **Iris seemed to not learn**
	```
	LR=0.01: loss=1.0987, accuracy=0.3000
	```
	All losses stuck at ~1.099 (-ln(1/3)). Thought it was a bug but I ran it with 500 epochs, it seemed to learn after that.