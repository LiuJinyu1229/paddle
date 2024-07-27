---
A paddlepaddle implementation for paper "Feature Learning based Deep Supervised Hashing with Pairwise Labels"
---

## Statement:
As pytorch doesn't provide pretrained VGG-F model, unlike original DPSH [paper](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf), we use pretrained Alexnet or pretrained VGG-11 for feature learning part instead of pretrained VGG-F.

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo:
```python
train:
$ python DPSH.py --train
test:
$ python DPSH.py
```

## Dataset
- mir25_crossmodal.h5
