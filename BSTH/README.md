## BSTH
The source code of "Bit-aware Semantic Transformer Hashing for Multi-modal Retrieval." (Accepted by SIGIR 2022)

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo
```
train:
python train_transformer.py --train
test:
python train_transformer.py
```

## Baselines
The baseline codes can be referred in my another repository.  
https://github.com/BMC-SDNU/Multi-Modal-Hashing-Retrieval

## Reference
  @inproceedings{BSTH2022,   
  &nbsp;&nbsp;&nbsp;&nbsp;title={Bit-aware Semantic Transformer Hashing for Multi-modal Retrieval},   
  &nbsp;&nbsp;&nbsp;&nbsp;author={Tan, Wentao and Zhu, Lei and Guan, Weili and Li, Jingjing and Cheng, Zhiyong},   
  &nbsp;&nbsp;&nbsp;&nbsp;booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},   
  &nbsp;&nbsp;&nbsp;&nbsp;pages={982--991},   
  &nbsp;&nbsp;&nbsp;&nbsp;year={2022}   
 }
