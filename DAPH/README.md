# DAPH

## Introduction
The implementation of "Data-Aware Proxy Hashing for Cross-modal Retrieval"

If there are not corresponding proxy hash codes in ./data_aware_proxy_codes, please generate the proxy hash codes by:
```python 
python DAPH_proxy_code.py
```

## Requirements

- Python 3.8.18
- paddlepaddle 2.1.0
- cuda 11.2
- cudnn 8.2.1

## Demo
```
train:
python DAPH.py --train
test:
python DAPH.py
```

## Dataset
`iaprtc.h5`
