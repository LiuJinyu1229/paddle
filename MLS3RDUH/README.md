# MLS3RDUH
paddlepaddle implementation for paper “MLS^3RDUH: Deep Unsupervised Hashing via Manifold based Local Semantic Similarity Structure Reconstructing”

## Brief Introduction
This paper introduces a novel unsupervised deep hashing method called MLS3RDUH, which reconstructs the local semantic similarity structure using manifold and cosine similarity between data points. A new similarity matrix is defined, and a novel log-cosh hashing loss function is used to optimize the hashing network, resulting in improved retrieval performance. Experimental results on three datasets demonstrate that MLS3RDUH outperforms state-of-the-art baselines, making it a significant contribution to unsupervised hashing methods.

## pretrain_loading
- 下载文件final_sim.npy至主目录
- 下载文件alexnet_flickr_features4096_labels.pkl至文件夹\features\下

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo
```python
train:
$ python MLS3RDUH.py --train
test:
$ python MLS3RDUH.py
```

## Dataset
- mir25_crossmodal.h5
