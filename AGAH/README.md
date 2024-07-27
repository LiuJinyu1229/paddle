# AGAH

A paddlepaddle implementation for paper [Adversary Guided Asymmetric Hashing for Cross-Modal Retrieval](https://dl.acm.org/citation.cfm?doid=3323873.3325045) (ICMR 2019 Best Student Paper).

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo
```
train:
$ python main.py train
test:
$ python main.py test
```

## Dataset
- FLICKR.mat
- imagenet-vgg-f.mat

## Citing AGAH

```
@inproceedings{gu2019adversary,
  title={Adversary Guided Asymmetric Hashing for Cross-Modal Retrieval},
  author={Gu, Wen and Gu, Xiaoyan and Gu, Jingzi and Li, Bo and Xiong, Zhi and Wang, Weiping},
  booktitle={Proceedings of the 2019 on International Conference on Multimedia Retrieval},
  pages={159--167},
  year={2019},
  organization={ACM}
}
```
