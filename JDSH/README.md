# JDSH

paddlepaddle implementataion for Joint-modal Distribution-based Similarity Hashing for Large-scale Unsupervised Deep Cross-modal Retrieval

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
- mir25_crossmodal.h5

## Citation
```
@inproceedings{JDSH,
    author={Liu, Song and Qian, Shengsheng and Guan, Yang and Zhan, Jiawei and Ying, Long},
    title={Joint-modal Distribution-based Similarity Hashing for Large-scale Unsupervised Deep Cross-modal Retrieval},
    booktitle = {SIGIR},
    year = {2020}
}

@inproceedings{DJSRH,
    author={Su, Shupeng and Zhong, Zhisheng and Zhang, Chao},
    title={Deep Joint-Semantics Reconstructing Hashing for Large-scale Unsupervised Deep Cross-modal Retrieval},
    booktitle = {ICCV},
    year = {2019}
}
```
