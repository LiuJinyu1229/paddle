
---
#  Source code for ADSH-AAAI2018 [paddlepaddle Version]
---
## Brief Introduction
This package contains the code for paper Asymmetric Deep Supervised Hashing on AAAI-2018. We only carry out experiment on CIFAR-10 and NUS-WIDE datasets. And we utilize pre-trained ResNet-50 for feature learning rather CNN-F in our original paper. Please note that the results for paper is based on MatConvNet version.

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo
```python
train:
$ python ADSH.py --train
test:
$ python ADSH.py
```

## Dataset
- mir25_crossmodal.h5
