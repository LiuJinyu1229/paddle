import paddle
import math
import os
import h5py
import numpy as np
ROOT = '/home1/ljy/dataset/'
if not os.path.exists(ROOT):
    raise Exception('The ROOT path is error.')
paths = {'flickr': ROOT + 'mir_cnn_twt.mat', 'nuswide': ROOT +
    'nus_cnn_twt.mat', 'coco': ROOT + 'coco_cnn_twt_2014.mat'}


def load_data(DATANAME, alpha_train=0.0, beta_train=0.5, alpha_query=0.0,
    beta_query=0.5):
    data = h5py.File(paths[DATANAME], 'r')
    I_tr = paddle.to_tensor(data=data['I_tr'][:].T, dtype='float32')
    T_tr = paddle.to_tensor(data=data['T_tr'][:].T, dtype='float32')
    L_tr = paddle.to_tensor(data=data['L_tr'][:].T, dtype='float32')
    I_db = paddle.to_tensor(data=data['I_db'][:].T, dtype='float32')
    T_db = paddle.to_tensor(data=data['T_db'][:].T, dtype='float32')
    L_db = paddle.to_tensor(data=data['L_db'][:].T, dtype='float32')
    I_te = paddle.to_tensor(data=data['I_te'][:].T, dtype='float32')
    T_te = paddle.to_tensor(data=data['T_te'][:].T, dtype='float32')
    L_te = paddle.to_tensor(data=data['L_te'][:].T, dtype='float32')
    complete_data = {'I_tr': I_tr, 'T_tr': T_tr, 'L_tr': L_tr, 'I_db': I_db,
        'T_db': T_db, 'L_db': L_db, 'I_te': I_te, 'T_te': T_te, 'L_te': L_te}
    train_missed_data = construct_missed_data(I_tr, T_tr, L_tr, alpha=
        alpha_train, beta=beta_train)
    query_missed_data = construct_missed_data(I_te, T_te, L_te, alpha=
        alpha_query, beta=beta_query)
    return complete_data, train_missed_data, query_missed_data


def construct_missed_data(I_tr, T_tr, L_tr, alpha=0.0, beta=0.5):
    number = I_tr.shape[0]
    dual_size = math.ceil(number * (1 - alpha))
    only_image_size = math.floor(number * alpha * beta)
    only_text_size = number - dual_size - only_image_size
    print('Dual size: %d, Oimg size: %d, Otxt size: %d' % (dual_size, only_image_size, only_text_size))
    random_idx = np.random.permutation(number)

    dual_index = paddle.to_tensor(random_idx[:dual_size].tolist())
    only_image_index = paddle.to_tensor(random_idx[dual_size:dual_size+only_image_size].tolist())
    only_text_index = paddle.to_tensor(random_idx[dual_size+only_image_size:dual_size+only_image_size+only_text_size].tolist())

    # import ipdb
    # ipdb.set_trace()

    I_dual_img = paddle.index_select(I_tr, dual_index, axis=0)
    I_dual_txt = paddle.index_select(T_tr, dual_index, axis=0)
    I_dual_label = paddle.index_select(L_tr, dual_index, axis=0)

    if len(only_image_index) == 0:
        I_oimg = paddle.empty(shape=[0, I_tr.shape[1]], dtype='float64')
        I_oimg_label = paddle.empty(shape=[0, L_tr.shape[1]], dtype='float64')
    else:
        I_oimg = paddle.index_select(I_tr, only_image_index, axis=0)
        I_oimg_label = paddle.index_select(L_tr, only_image_index, axis=0)

    if len(only_text_index) == 0:
        I_otxt = paddle.empty(shape=[0, T_tr.shape[1]], dtype='float64')
        I_otxt_label = paddle.empty(shape=[0, L_tr.shape[1]], dtype='float64')
    else:
        I_otxt = paddle.index_select(T_tr, only_text_index, axis=0)
        I_otxt_label = paddle.index_select(L_tr, only_text_index, axis=0)

    _dict = {'I_dual_img': I_dual_img, 'I_dual_txt': I_dual_txt, 'I_dual_label': I_dual_label, 
             'I_oimg': I_oimg, 'I_oimg_label': I_oimg_label, 'I_otxt': I_otxt, 'I_otxt_label': I_otxt_label}
    return _dict


class CoupledData(paddle.io.Dataset):

    def __init__(self, img_feature, txt_feature):
        self.img_feature = img_feature
        self.txt_feature = txt_feature
        self.length = self.img_feature.shape[0]

    def __getitem__(self, item):
        return self.img_feature[(item), :], self.txt_feature[(item), :]

    def __len__(self):
        return self.length


class TrainCoupledData(paddle.io.Dataset):

    def __init__(self, img_feature, txt_feature, label):
        self.img_feature = img_feature
        self.txt_feature = txt_feature
        self.label = label
        self.length = self.img_feature.shape[0]

    def __getitem__(self, item):
        return self.img_feature[(item), :], self.txt_feature[(item), :
            ], self.label[(item), :]

    def __len__(self):
        return self.length
