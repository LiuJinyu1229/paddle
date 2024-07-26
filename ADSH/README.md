
---
#  Source code for ADSH-AAAI2018 [paddlepaddle Version]
---
## Introduction
### 1. Brief Introduction
This package contains the code for paper Asymmetric Deep Supervised Hashing on AAAI-2018. We only carry out experiment on CIFAR-10 and NUS-WIDE datasets. And we utilize pre-trained ResNet-50 for feature learning rather CNN-F in our original paper. Please note that the results for paper is based on MatConvNet version.
### 2. Running Environment
```python
Python 3.7.2
paddlepaddle 2.5.2
```
### 3. Run Demo
```python
train:
python ADSH.py --train

test:
python ADSH.py
```
