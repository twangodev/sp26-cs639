#!/bin/bash
mkdir -p data

# Iris
curl -o data/iris.data https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

# California Housing
curl -o data/housing.csv https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv

# MNIST
curl -o data/train-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -o data/train-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -o data/t10k-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -o data/t10k-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# Unzip MNIST
gunzip -f data/*.gz