import cv2
import h5py
import numpy as np
import scipy.io as scio
import paddle
from PIL import Image
from paddle.vision import transforms

import settings
np.random.seed(1)
if settings.DATASET == "WIKI":

    label_set = scio.loadmat(settings.LABEL_DIR)
    test_txt = np.array(label_set['T_te'], dtype=np.float)
    train_txt = np.array(label_set['T_tr'], dtype=np.float)

    test_label = []
    with open(settings.TEST_LABEL, 'r') as f:
        for line in f.readlines():
            test_label.extend([int(line.split()[-1]) - 1])

    test_img_name = []
    with open(settings.TEST_LABEL, 'r') as f:
        for line in f.readlines():
            test_img_name.extend([line.split()[1]])

    train_label = []
    with open(settings.TRAIN_LABEL, 'r') as f:
        for line in f.readlines():
            train_label.extend([int(line.split()[-1]) - 1])

    train_img_name = []
    with open(settings.TRAIN_LABEL, 'r') as f:
        for line in f.readlines():
            train_img_name.extend([line.split()[1]])

    wiki_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    wiki_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_feat_len = train_txt.shape[1]


    class WIKI(paddle.io.Dataset):

        def __init__(self, root, transform=None, target_transform=None, train=True):
            self.root = root
            self.transform = transform
            self.target_transform = target_transform
            self.f_name = ['art', 'biology', 'geography', 'history', 'literature', 'media', 'music', 'royalty', 'sport',
                           'warfare']
            
            if train:
                self.label = train_label
                self.img_name = train_img_name
                self.txt = train_txt
            else:
                self.label = test_label
                self.img_name = test_img_name
                self.txt = test_txt

        def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is index of the target class.
            """

            path = self.root + '/' + self.f_name[self.label[index]] + '/' + self.img_name[index] + '.jpg'
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            target = self.label[index]
            txt = self.txt[index]
            # print(len(txt), txt, path, target, '###\n')
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, txt, target, index

        def __len__(self):
            return len(self.label)