# DJSRH
***********************************************************************************************************

This repository is for ["Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Deep_Joint-Semantics_Reconstructing_Hashing_for_Large-Scale_Unsupervised_Cross-Modal_Retrieval_ICCV_2019_paper.pdf) 

(to appear in ICCV 2019, Oral)

By Shupeng Su\*, [Zhisheng Zhong](https://zzs1994.github.io)\*, Chao Zhang (\* Authors contributed equally).

## Introduction

Cross-modal hashing encodes the multimedia data into a common binary hash space in which the correlations among the samples from different modalities can be effectively measured. Deep cross-modal hashing further improves the retrieval performance as the deep neural networks can generate more semantic relevant features and hash codes. In this paper, we study the unsupervised deep cross-modal hash coding and propose Deep JointSemantics Reconstructing Hashing (DJSRH), which has the following two main advantages. First, to learn binary codes that preserve the neighborhood structure of the original data, DJSRH constructs a novel joint-semantics affinity matrix which elaborately integrates the original neighborhood information from different modalities and accordingly is capable to capture the latent intrinsic semantic affinity for the input multi-modal instances. Second, DJSRH later trains the networks to generate binary codes that maximally reconstruct above joint-semantics relations via the proposed reconstructing framework, which is more competent for the batch-wise training as it reconstructs the specific similarity value unlike the common Laplacian constraint merely preserving the similarity order. Extensive experiments demonstrate the significant improvement by DJSRH in various cross-modal retrieval tasks.

<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/DJRSH.png" width="90%" height="90%"></div align=center>

***********************************************************************************************************

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo
```
train:
python demo_DJSRH.py --train
test:
python demo_DJSRH.py
```

## Dataset
- mir_cnn_twt.mat

## Citation

If you find this code useful, please cite our paper:

	@inproceedings{su2019deep,
		title={Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval},
		author={Shupeng Su, Zhisheng Zhong, Chao Zhang},
		booktitle={International Conference on Computer Vision},
		year={2019}
	}

All rights are reserved by the authors.
***********************************************************************************************************
