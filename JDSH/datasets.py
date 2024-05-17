import paddle
from PIL import Image
from args import config
from paddle.vision.transforms import transforms
import h5py

data_file = h5py.File(config.DATA_DIR, 'r')

mir_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

mir_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class MIRFlickr(paddle.io.Dataset):
    def __init__(self, transform=None, target_transform=None, train=True, database=False):
        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.labels = data_file['train_L']
            self.img = data_file['train_data']
            self.txt = data_file['train_y']
        elif database:
            self.labels = data_file['dataset_L']
            self.img = data_file['data_set']
            self.txt = data_file['dataset_y']
        else:
            self.labels = data_file['test_L']
            self.img = data_file['test_data']
            self.txt = data_file['test_y']

    def __getitem__(self, index):
        txt = self.txt[index]
        img = self.img[index]
        label = self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, txt, label, index

    def __len__(self):
        return self.img.shape[0]
