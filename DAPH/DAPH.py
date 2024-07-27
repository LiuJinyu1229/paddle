import pickle
import os
import argparse
import logging
import paddle
import time
import numpy as np
import paddle.optimizer as optim
import paddle.vision.transforms as transforms
import h5py
from datetime import datetime
from paddle import to_tensor
from paddle.io import DataLoader
import paddle.nn.functional as F
import dataset_my as dp
import utils.hash_model as hash_models
import utils.calc_hr as calc_hr
import paddle.nn as nn

paddle.device.set_device('gpu:6')

def GenerateCode(model_hash, model_text, data_loader, num_data, bit, k=0):
    B = np.zeros((num_data, bit), dtype=np.float32)
    Bt = np.zeros((num_data, bit), dtype=np.float32)
    kk = 1
    for iter, data in enumerate(data_loader, 0):
        data_img, data_text, _,  data_ind = data

        data_text = paddle.to_tensor(data_text.astype('float32')).cuda()
        data_img = paddle.to_tensor(data_img).cuda()
        if k == 0:
            out = model_hash(data_img)
            text_out = model_text(data_text)
            if kk:
                # print(out.item())
                kk = 0
            B[data_ind.numpy(), :] = paddle.sign(out)
            Bt[data_ind.numpy(), :] = paddle.sign(text_out)
    return B,  Bt

def loss_function(hash_out, anchor_code, positive_select, negtive_select, sigma, gamma, eps=1e-5):
    logit_ii = paddle.matmul(hash_out, paddle.transpose(anchor_code, [1, 0]))
    code_length = hash_out.shape[1]
    the_batch = hash_out.shape[0]
    our_logit_ii = paddle.exp((logit_ii / float(code_length) - sigma) * gamma) * positive_select
    mu_logit_ii = (paddle.exp(logit_ii * gamma / float(code_length)) * negtive_select).sum(1).reshape([-1, 1]).expand([the_batch, anchor_code.shape[0]]) + our_logit_ii
    xx = (mu_logit_ii < eps).astype('float32')
    loss_ii = - ((paddle.log((our_logit_ii + (1 - positive_select)*eps) / (mu_logit_ii + xx.detach() * eps)) * positive_select).sum(1) / positive_select.sum(1)).sum() / the_batch

    return loss_ii

def loss_mean_function(hash_out, anchor_code, positive_select, negtive_select, sigma, gamma, eps=1e-5):
    logit_ii = paddle.matmul(hash_out, paddle.transpose(anchor_code, [1, 0]))
    code_length = hash_out.shape[1]
    the_batch = hash_out.shape[0]
    our_logit_ii = paddle.exp(((logit_ii * positive_select).sum(1) / (positive_select.sum(1) * float(code_length)) - sigma) * gamma)
    mu_logit_ii = (paddle.exp(logit_ii * gamma / float(code_length)) * negtive_select).sum(1) + our_logit_ii
    loss_ii = - (paddle.log(our_logit_ii / mu_logit_ii)).sum() / the_batch
    return loss_ii

def loss_mean_norm_function(hash_out, anchor_code, positive_select, norm_positive_select, negtive_select, sigma, gamma, eps=1e-5):
    logit_ii = paddle.matmul(hash_out, paddle.transpose(anchor_code, [1, 0]))
    code_length = hash_out.shape[1]
    our_logit_ii = paddle.exp(((logit_ii * norm_positive_select / float(code_length)).sum(1) - sigma) * gamma)
    mu_logit_ii = (paddle.exp(logit_ii * gamma / float(code_length)) * negtive_select).sum(1) + our_logit_ii
    no_zero_positive = (positive_select.sum(1) > 0.1).astype('float32')
    xx = (mu_logit_ii < eps).astype('float32')
    loss = - (paddle.log(our_logit_ii / (mu_logit_ii + xx * eps) + (1 - no_zero_positive) * eps) * no_zero_positive).sum() / no_zero_positive.sum()

    return loss


def DAPH_Algo(opt):
    data_set = opt.dataset
    code_length = opt.bit
    bit = opt.bit

    batch_size = 128
    epochs = 100 # 220
    weight_decay = 10 ** -5
    model_name = 'alexnet'

    gamma = opt.gamma
    sigma = opt.sigma
    file_name = './checkpoint/DAPH_' + data_set + '_' + str(code_length) + '.pdparams'
    ### data processing

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    all_dta = h5py.File(opt.data_path)
    text_len = np.asarray(all_dta['test_y']).shape[1]
    dset_database = dp.DatasetPorcessing_h5(
        np.asarray(all_dta['data_set']), np.asarray(all_dta['dataset_y']), np.asarray(all_dta['dataset_L']),
        transformations)
    dset_train = dp.DatasetPorcessing_h5(
        np.asarray(all_dta['train_data']), np.asarray(all_dta['train_y']), np.asarray(all_dta['train_L']),
        transformations)
    dset_test = dp.DatasetPorcessing_h5(
        np.asarray(all_dta['test_data']), np.asarray(all_dta['test_y']), np.asarray(all_dta['test_L']),
        transformations)

    learning_ratet = 0.005
    learning_ratei = 0.005
    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)
    database_loader = DataLoader(dset_database,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True
                              )
    test_loader = DataLoader(dset_test,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    database_labels = dset_database.labels
    test_labels = dset_test.labels
    train_labels = dset_train.labels


    label_size = test_labels.shape
    nclass = label_size[1]
    fp = './data_aware_proxy_codes/' + data_set + '_' + str(code_length) + '.pkl'
    with open(fp, 'rb') as f:
        all_train = pickle.load(f)
        anchor_code = paddle.sign(paddle.to_tensor(all_train['image_code']))
        anchor_text_code = paddle.sign(paddle.to_tensor(all_train['text_code']))
        anchor_label = paddle.to_tensor(all_train['label'])
        label_code = paddle.sign(paddle.to_tensor(all_train['class_code']))

    hash_model = hash_models.HASH_Net(model_name, code_length)
    text_model = hash_models.TxtNet(text_len, code_length)
    optimizer_hash = paddle.optimizer.SGD(parameters=hash_model.parameters(), learning_rate=learning_ratei, weight_decay=weight_decay)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=learning_ratei, step_size=300, gamma=0.3, last_epoch=-1, verbose=False)
    optimizer_text = paddle.optimizer.SGD(parameters=text_model.parameters(), learning_rate=learning_ratet, weight_decay=weight_decay)
    scheduler_text = paddle.optimizer.lr.StepDecay(learning_rate=learning_ratet, step_size=300, gamma=0.3, last_epoch=-1, verbose=False)

    eps = 1e-5
    if opt.train:
        print('start training...')
        for epoch in range(epochs):
            hash_model.train()
            text_model.eval()
            scheduler.step()
            epoch_loss = 0.0
            ## training epoch
            for iter, traindata in enumerate(train_loader, 0):
                train_img, train_text, train_label, batch_ind = traindata
                train_label = paddle.squeeze(train_label)
                train_img = paddle.to_tensor(train_img).cuda()
                train_label = paddle.to_tensor(train_label.astype('float32')).cuda()

                hash_out = hash_model(train_img)

                simxall = paddle.matmul(train_label, paddle.transpose(anchor_label, [1, 0]))
                positive_select = 1. * (simxall > (paddle.sum(train_label, axis=1).reshape([-1, 1]) - 0.2)) * (simxall > (paddle.sum(anchor_label, axis=1).reshape([1, -1]) - 0.2))
                negtive_select = 1 - 1. * (simxall > 0)
                partial_positive_select = 1. - positive_select - negtive_select

                norm_partial_positive_select = partial_positive_select * simxall / (paddle.sum(simxall, axis=1).reshape([-1, 1]) + eps)

                loss_i = loss_function(hash_out, anchor_code, positive_select, negtive_select, sigma, gamma)
                loss_t = loss_function(hash_out, anchor_text_code, positive_select, negtive_select, sigma, gamma)
                loss_pi = loss_mean_norm_function(hash_out, anchor_code, partial_positive_select, norm_partial_positive_select, negtive_select, sigma, gamma)
                loss_pt = loss_mean_norm_function(hash_out, anchor_text_code, partial_positive_select, norm_partial_positive_select, negtive_select, sigma, gamma)
                loss_l = loss_mean_function(hash_out, label_code, train_label, 1- train_label, sigma, gamma)
                loss_all = loss_i + loss_t + loss_l + (loss_pi + loss_pt) * opt.eta

                optimizer_hash.clear_grad()
                loss_all.backward()
                optimizer_hash.step()
                epoch_loss += loss_all.numpy().item()
            print(
                '[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' %
                (epoch + 1, epochs, epoch_loss / len(train_loader)))

            hash_model.eval()
            text_model.train()
            scheduler_text.step()
            epoch_loss = 0.0
            for iter, traindata in enumerate(train_loader, 0):
                train_img, train_text, train_label, batch_ind = traindata
                train_label = paddle.squeeze(train_label)
                train_label = paddle.to_tensor(train_label.astype('float32'))
                train_text = paddle.to_tensor(train_text.astype('float32'))

                hash_out = text_model(train_text)

                simxall = paddle.matmul(train_label, paddle.transpose(anchor_label, [1, 0]))
                positive_select = 1. * (simxall > (paddle.sum(train_label, axis=1).reshape([-1, 1]) - 0.2))
                negtive_select = 1 - 1. * (simxall > 0)
                partial_positive_select = (1. - negtive_select) * (simxall < (paddle.sum(train_label, axis=1).reshape([-1, 1]) - 0.2))

                norm_partial_positive_select = partial_positive_select * simxall / (paddle.sum(simxall, axis=1).reshape([-1, 1]) + eps)

                loss_i = loss_function(hash_out, anchor_code, positive_select, negtive_select, sigma, gamma)
                loss_t = loss_function(hash_out, anchor_text_code, positive_select, negtive_select, sigma, gamma)
                loss_pi = loss_mean_norm_function(hash_out, anchor_code, partial_positive_select, norm_partial_positive_select, negtive_select, sigma, gamma)
                loss_pt = loss_mean_norm_function(hash_out, anchor_text_code, partial_positive_select, norm_partial_positive_select, negtive_select, sigma, gamma)
                loss_l = loss_mean_function(hash_out, label_code, train_label, 1 - train_label, sigma, gamma)
                loss_all = loss_i + loss_t + loss_l + (loss_pi + loss_pt) * opt.eta

                optimizer_text.clear_gradients()
                loss_all.backward()
                optimizer_text.step()
                epoch_loss += loss_all.item()
            print(
                '[Train Phase][Epocht: %3d/%3d][Loss_text: %3.5f]' %
                (epoch + 1, epochs, epoch_loss / len(train_loader)))
        paddle.save({'hash_model': hash_model.state_dict(), 'text_model': text_model.state_dict()}, file_name)
    else:
        checkpoint = paddle.load(file_name)
        hash_model.set_state_dict(checkpoint['hash_model'])
        text_model.set_state_dict(checkpoint['text_model'])

    print('start testing...')
    hash_model.eval()
    text_model.eval()
    qi, qt = GenerateCode(hash_model, text_model, test_loader, num_test, bit)
    ri, rt = GenerateCode(hash_model, text_model, database_loader, num_database, bit)
    map_it = calc_hr.calc_map(qi, rt, test_labels, database_labels)
    map_ti = calc_hr.calc_map(qt, ri, test_labels, database_labels)
    print('map_i2t:', map_it, 'map_t2i:', map_ti)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', default=1.0, type=float, help='eta')
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--bit', default=64, type=int, help='hash code length')
    parser.add_argument('--gamma', default=20, type=int, help='gamma')
    parser.add_argument('--sigma', default=0.2, type=float, help='sigma')
    parser.add_argument('--dataset', default='iaprtc', type=str, help='dataset name')
    parser.add_argument('--data_path', default='/home1/ljy/dataset/iaprtc.h5', type=str, help='dataset path')

    opt = parser.parse_args()
    DAPH_Algo(opt)
