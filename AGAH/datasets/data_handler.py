import os
import numpy as np
import h5py
from scipy.io import loadmat

def preprocess(x, mean, std):
    mean, std = np.array(mean), np.array(std)
    return (x - mean.reshape(3, 1, 1)) / (std.reshape(3, 1, 1) + 1e-5)


def load_data(path, type='flickr25k'):
    if type == 'flickr25k':
        return load_flickr25k(path)


def load_flickr25k(path):
    data_file = h5py.File(path)
    images = data_file['images'][:]
    tags = data_file['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels

def data_enhance(images, tags, labels, c=0.5):
    num = images.shape[0]
    ind1 = np.random.permutation(num // 2)
    ind2 = np.random.permutation(num // 2)
    inhanced_imgs = c * images[ind1] + (1 - c) * images[ind2]
    inhanced_tags = c * tags[ind1] + (1 - c) * tags[ind2]
    inhanced_labels = c * labels[ind1] + (1 - c) * labels[ind2]
    new_imgs = np.concatenate((images, inhanced_imgs))
    new_tags = np.concatenate((tags, inhanced_tags))
    new_labels = np.concatenate((labels, inhanced_labels))
    return new_imgs, new_tags, new_labels


def load_pretrain_model(path):
    return loadmat(path)

