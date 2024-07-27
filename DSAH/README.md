# DSAH
> Source code of our ICMR 2020 paper "[Deep Semantic-Alignment Hashing for Unsupervised Cross-Modal Retrieval](https://dl.acm.org/doi/abs/10.1145/3372278.3390673)"

## Introduction

Deep hashing methods have achieved tremendous success in cross-modal retrieval, due to its low storage consumption and fast retrieval speed. In real cross-modal retrieval applications, it's hard to obtain label information. Recently, increasing attention has been paid to unsupervised cross-modal hashing. However, existing methods fail to exploit the intrinsic connections between images and their corresponding descriptions or tags (text modality). In this paper, we propose a novel Deep Semantic-Alignment Hashing (DSAH) for unsupervised cross-modal retrieval, which sufficiently utilizes the co-occurred image-text pairs. DSAH explores the similarity information of different modalities and we elaborately design a semantic-alignment loss function, which elegantly aligns the similarities between features with those between hash codes. Moreover, to further bridge the modality gap, we innovatively propose to reconstruct features of one modality with hash codes of the other one. Extensive experiments on three cross-modal retrieval datasets demonstrate that DSAH achieves the state-of-the-art performance.
![Framework](https://github.com/idejie/pics/raw/master/WX20200627-190524.png)

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo
```
train:
$ python main.py --train
test:
$ python main.py
```

## Dataset
- wikipedia
## Citation
If you find this code useful, please cite our paper:
```bibtex
@inproceedings{10.1145/3372278.3390673,
author = {Yang, Dejie and Wu, Dayan and Zhang, Wanqian and Zhang, Haisu and Li, Bo and Wang, Weiping},
title = {Deep Semantic-Alignment Hashing for Unsupervised Cross-Modal Retrieval},
year = {2020},
isbn = {9781450370875},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3372278.3390673},
doi = {10.1145/3372278.3390673},
booktitle = {Proceedings of the 2020 International Conference on Multimedia Retrieval},
pages = {44–52},
numpages = {9},
keywords = {cross-modal hashing, cross-media retrieval, semantic-alignment},
location = {Dublin, Ireland},
series = {ICMR ’20}
}
```
All rights are reserved by the authors.
## References
- [zzs1994/DJSRH](https://github.com/zzs1994/DJSRH)
