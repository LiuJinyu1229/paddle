import paddle
import h5py
from PIL import Image
import numpy as np
import settings
import paddle.vision.transforms as transforms

all_data = h5py.File(settings.DIR, 'r')

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

txt_feat_len = all_data['T_tr'].shape[0]

class MY_DATASET(paddle.io.Dataset):

    def __init__(self, transform=None, target_transform=None, train=True,
        database=False):
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.train_labels = all_data['L_tr'][:].T
            self.txt = all_data['T_tr'][:].T
            # self.images = all_data['I_tr'][:].transpose(3, 0, 1, 2)
            self.images = all_data['I_tr'][:].T
        elif database:
            self.train_labels = all_data['L_db'][:].T
            self.txt = all_data['T_db'][:].T
            # self.images = all_data['I_db'][:].transpose(3, 0, 1, 2)
            self.images = all_data['I_db'][:].T
        else:
            self.train_labels = all_data['L_te'][:].T
            self.txt = all_data['T_te'][:].T
            # self.images = all_data['I_te'][:].transpose(3, 0, 1, 2)
            self.images = all_data['I_te'][:].T

    def __getitem__(self, index):
        img, target = self.images[index], self.train_labels[index]
        img = Image.fromarray(img)
        txt = self.txt[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, txt, target, index

    def __len__(self):
        return len(self.train_labels)
