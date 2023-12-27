import paddle
import os
import numpy as np
from PIL import Image
from paddle.io import Dataset
import pickle
import random

class DatasetPorcessing_h5(Dataset):
    def __init__(self, train_data, train_y, train_label, transform=None, is_train=False):
        self.train_data = train_data
        self.is_train = is_train
        self.transform = transform
        self.train_y = train_y
        print(train_y.shape, train_data.shape)
        self.labels = train_label

    def __getitem__(self, index):
        img = Image.fromarray(self.train_data[index])
        if self.transform is not None:
            img1 = self.transform(img)
        label = self.labels[index]
        y_vector = self.train_y[index]
        # while label.sum() < 1:
        #     index = random.randint(0, len(self.train_data) - 1)
        #     img = Image.fromarray(self.train_data[index])
        #     if self.transform is not None:
        #         img1 = self.transform(img)
        #     label = self.labels[index]
        #     y_vector = paddle.to_tensor(self.train_y[index]).astype('float32')
        return img1, y_vector, label, index

    def __len__(self):
        return self.labels.shape[0]