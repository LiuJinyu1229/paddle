import paddle
from PIL import Image
from paddle.io import Dataset
import h5py
from PIL import Image

import paddle
import h5py
from paddle.vision.transforms import transforms

# train_transform = vision.transforms.Compose([
#     vision.transforms.Resize(256),
#     vision.transforms.RandomCrop(224),
#     vision.transforms.RandomHorizontalFlip(), 
#     vision.transforms.ToTensor(), 
#     vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# test_transform = vision.transforms.Compose([
#     vision.transforms.Resize(256), 
#     vision.transforms.CenterCrop(224), 
#     vision.transforms.ToTensor(), 
#     vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
dataset = 'FLICKR-25K'
if dataset == 'nus':
    data_dir = '/home1/ljy/dataset/nus.h5'
elif dataset == 'coco':
    data_dir = '/home1/ljy/dataset/coco_pslh.h5'
elif dataset == 'FLICKR-25K':
    # data_dir = '/home1/ljy/dataset/mir25_crossmodal.h5'
    data_dir = '/home1/ljy/dataset/mir_cnn_twt.mat'
data = h5py.File(data_dir, 'r')
print(data.keys())

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
            # self.label = paddle.to_tensor(data["train_L"][:]).astype('float32')
            # self.label = data["train_L"][:]
            # self.total_img = data["train_data"][:]
            self.label = data['L_tr'][:].T
            self.total_img = data['I_tr'][:].T
        elif database:
            # self.label = paddle.to_tensor(data["dataset_L"][:]).astype('float32')
            # self.label = data["dataset_L"][:]
            # self.total_img = data["data_set"][:]
            self.label = data['L_db'][:].T
            self.total_img = data['I_db'][:].T
        else:
            # self.label = paddle.to_tensor(data["test_L"][:]).astype('float32')
            # self.label = data["test_L"][:]
            # self.total_img = data["test_data"][:]
            self.label = data['L_te'][:].T
            self.total_img = data['I_te'][:].T
    
    def __getitem__(self, index):
        img_data = self.total_img[index]
        label = self.label[index]
        img = Image.fromarray(img_data)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.total_img)