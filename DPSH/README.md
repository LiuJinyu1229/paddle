---
A paddlepaddle implementation for paper "Feature Learning based Deep Supervised Hashing with Pairwise Labels"
---
### 1. Running Environment:
```python
Python 3.7.2
paddlepaddle 2.5.2
```
### 2. Statement:
As pytorch doesn't provide pretrained VGG-F model, unlike original DPSH [paper](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf), we use pretrained Alexnet or pretrained VGG-11 for feature learning part instead of pretrained VGG-F.
### 3. Demo:
```python
train:
python DPSH.py --train

test:
python DPSH.py
```
