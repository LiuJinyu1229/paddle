# Deep Adversarial Discrete Hashing for Cross-Modal Retrieval

## Introduction

This is the source code of ICMR 2020 paper "Deep Adversarial Discrete Hashing for Cross-Modal Retrieval".

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo
```
train:
python main.py --train
test:
python main.py
```

## Dataset
- FLICKR-25K.mat

## Note

Our codes were modified from the implementation of "Adversary Guided Asymmetric Hashing for Cross-Modal Retrieval", written by Wen Gu. Please cite the  two papers (AGAH and DADH) when you use the codes.

## Citing DADH & AGAH

```
@inproceedings{Bai2020,
  author={Cong Bai, Chao Zeng, Qing Ma, Jinglin Zhang and Shengyong Chen.},
  booktitle={Proceedings of the 2020 on International Conference on Multimedia Retrieval},
  pages={525-531},
  title={Deep Adversarial Discrete Hashing for Cross-Modal Retrieval},
  year={2020},
}
```
```
@inproceedings{Gu2019,
author = {Gu, Wen and Gu, Xiaoyan and Gu, Jingzi and Li, Bo and Xiong, Zhi and Wang, Weiping},
booktitle = {Proceedings of the ACM International Conference on Multimedia Retrieval},
pages = {159--167},
title = {{Adversary guided asymmetric hashing for cross-modal retrieval}},
year = {2019}
}
```
