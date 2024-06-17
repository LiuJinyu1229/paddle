# SCH
Source code for TPAMI paper "[Cross-Modal Hashing Method with Properties of Hamming Space: A New Perspective](https://ieeexplore.ieee.org/document/10506992/)"

## Datasets
Please refer to the provided [link](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_tensorflow/DCMH_tensorflow) to download the dataset, create a data folder and update data path in settings.py.

## Environment

`Python 3.7.2`
`paddlepaddle 2.5.2`

## Train model

You can directly run the file 
```
python train.py --Bit 16 --GID 0 --DS 0
```
to get the results.

## Evaluate the model

Modify the settings.py
```
EVAL = True
```

## Citation
If you find SCH useful in your research, please consider citing:

```
@article{hu2024cross,
  title={Cross-Modal Hashing Method with Properties of Hamming Space: A New Perspective},
  author={Hu, Zhikai and Cheung, Yiu-ming and Li, Mengke and Lan, Weichao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

