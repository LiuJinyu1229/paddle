import paddle
import os
import numpy as np
from PIL import Image
from paddle.io import Dataset
import pickle
import h5py
from paddle.vision.transforms import transforms

dataset = 'flickr'
if dataset == 'nuswide':
    data_dir = '/home1/ljy/dataset/nus_cnn_twt.mat'
elif dataset == 'coco':
    data_dir = '/home1/ljy/dataset/coco_cnn_twt.mat'
elif dataset == 'flickr':
    data_dir = '/home/zhangyh/code/ljy/dataset/mir25_crossmodal.h5'
data = h5py.File(data_dir, 'r')

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DatasetProcessing(paddle.io.Dataset):
    def __init__(self, transform=None, train=True, database=False):
        self.transform = transform
        if train:
            self.label = data['train_L']
            self.total_img = data['train_data']
        elif database:
            self.label = data['dataset_L']
            self.total_img = data['data_set']
        else:
            self.label = data['test_L']
            self.total_img = data['test_data']
    
    def __getitem__(self, index):
        img_data = self.total_img[index]
        label = self.label[index]
        img = Image.fromarray(img_data)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.total_img)
