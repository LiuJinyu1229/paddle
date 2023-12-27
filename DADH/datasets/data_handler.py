import os
import numpy as np
import h5py
import scipy.io as scio


def load_data(path, type='flickr'):
    if type == 'flickr':
        return load_flickr(path)
    else:
        return load_nus_wide(path)


def load_flickr(path):
    print(path)
    data_file = scio.loadmat(path)
    images = data_file['images'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['YAll'][:]
    labels = data_file['LAll'][:]
    # all_data = h5py.File(path, 'r')
    # images = all_data['images'][:]
    # images = (images - images.mean()) / images.std()
    # tags = all_data['YAll'][:]
    # labels = all_data['LAll'][:]
    # with h5py.File(path, 'r') as f:
    #     images = np.array(f['images'][:])
    #     images = (images - images.mean()) / images.std()
    #     tags = np.array(f['YAll'][:])
    #     labels = np.array(f['LAll'][:])
    return images, tags, labels


def load_nus_wide(path_dir):
    data_file = scio.loadmat(path_dir)
    images = data_file['image'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['text'][:]
    labels = data_file['label'][:]

    return images, tags, labels


def load_pretrain_model(path):
    return scio.loadmat(path)

