import paddle
import h5py
from collections import namedtuple
ROOT = '/home1/ljy/dataset/'
dataset_tuple = namedtuple('dataset_tuple', ['I_tr', 'T_tr', 'L_tr', 'I_db',
    'T_db', 'L_db', 'I_te', 'T_te', 'L_te'])
paths = {'flickr': ROOT + 'mir_cnn_twt.mat', 
        'nuswide': ROOT + 'nus_cnn_twt.mat', 
        'coco': ROOT + 'coco_cnn_twt.mat'}


def load_data(DATANAME):
    data = h5py.File(paths[DATANAME], 'r')
    I_tr = data['I_tr'][:].T
    T_tr = data['T_tr'][:].T
    L_tr = data['L_tr'][:].T
    I_db = data['I_db'][:].T
    T_db = data['T_db'][:].T
    L_db = data['L_db'][:].T
    I_te = data['I_te'][:].T
    T_te = data['T_te'][:].T
    L_te = data['L_te'][:].T
    return dataset_tuple(I_tr=I_tr, T_tr=T_tr, L_tr=L_tr, I_db=I_db, T_db=
        T_db, L_db=L_db, I_te=I_te, T_te=T_te, L_te=L_te)


class my_dataset(paddle.io.Dataset):

    def __init__(self, img_feature, txt_feature, label, **kwargs):
        self.img_feature = paddle.to_tensor(data=img_feature)
        self.txt_feature = paddle.to_tensor(data=txt_feature)
        self.label = paddle.to_tensor(data=label)
        self.length = self.img_feature.shape[0]
        self.training = False
        if 'B_tr' in kwargs.keys():
            print('Convert the dataset state to "Training"')
            self.training = True
            self.B_tr = kwargs['B_tr']

    def __getitem__(self, item):
        if self.training:
            return item, paddle.to_tensor(self.img_feature[(item), :]).cuda(), paddle.to_tensor(self.txt_feature[(item), :]).cuda(), paddle.to_tensor(self.label[(item), :]).cuda(), paddle.to_tensor(self.B_tr[(item), :]).cuda()
        else:
            return item, paddle.to_tensor(self.img_feature[(item), :]).cuda(), paddle.to_tensor(self.txt_feature[(item), :]).cuda(), paddle.to_tensor(self.label[(item), :]).cuda()
    
    def __len__(self):
        return self.length


class l2h_dataset(paddle.io.Dataset):

    def __init__(self, label_index_sequence, label):
        self.label_index_sequence = label_index_sequence
        self.label = label
        self.length = self.label.shape[0]

    def __getitem__(self, item):
        return self.label_index_sequence[(item), :], self.label[(item), :]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dset = load_data('nuswide')
