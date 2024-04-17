# MLS3RDUH
paddlepaddle implementation for paper “MLS^3RDUH: Deep Unsupervised Hashing via Manifold based Local Semantic Similarity Structure Reconstructing”

## Introduction
### 1. Brief Introduction
This paper introduces a novel unsupervised deep hashing method called MLS3RDUH, which reconstructs the local semantic similarity structure using manifold and cosine similarity between data points. A new similarity matrix is defined, and a novel log-cosh hashing loss function is used to optimize the hashing network, resulting in improved retrieval performance. Experimental results on three datasets demonstrate that MLS3RDUH outperforms state-of-the-art baselines, making it a significant contribution to unsupervised hashing methods.
### 2. Running Environment
```python
Python 3.7.2
paddlepaddle 2.5.2
```
### 3. Run Demo
```python
$ python MLS3RDUH.py
```
