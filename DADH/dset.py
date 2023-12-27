import h5py
import paddle
from paddle.vision.transforms import Compose, Resize, RandomCrop, Normalize, RandomHorizontalFlip, ToTensor

from config import opt

all_data = h5py.File('/home1/ljy/dataset/mir_cnn_twt.mat', 'r')

train_transform = Compose([
    RandomHorizontalFlip(),
    Resize(256),
    RandomCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MY_DATASET(paddle.io.Dataset):
    def __init__(self, opt, train=True, database=False):
        if train:
            self.labels = paddle.to_tensor(all_data['L_tr'][:].T, dtype='float32')
            self.txt = paddle.to_tensor(all_data['T_tr'][:].T, dtype='float32')
            self.images = paddle.to_tensor(all_data['I_tr'][:].T, dtype='float32')
        elif database:
            self.labels = paddle.to_tensor(all_data['L_db'][:].T, dtype='float32')
            self.txt = paddle.to_tensor(all_data['T_db'][:].T, dtype='float32')
            self.images = paddle.to_tensor(all_data['I_db'][:].T, dtype='float32')
        else:
            self.labels = paddle.to_tensor(all_data['L_te'][:].T, dtype='float32')
            self.txt = paddle.to_tensor(all_data['T_te'][:].T, dtype='float32')
            self.images = paddle.to_tensor(all_data['I_te'][:].T, dtype='float32')

    def __getitem__(self, index):
        img = self.images[index]
        target = self.labels[index]
        txt = self.txt[index]

        return img, txt, target, index

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)