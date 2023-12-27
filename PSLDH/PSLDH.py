import pickle
import os
import argparse
import logging
import time
import numpy as np
import paddle.optimizer as optim
import paddle.vision.transforms as transforms
import paddle

from datetime import datetime
from paddle.io import DataLoader

import utils.data_processing as dp
import utils.hash_model as image_hash_model
import utils.label_hash_model as label_hash_model
import utils.calc_hr as calc_hr
import paddle.nn as nn

paddle.device.set_device('gpu:7')

def load_label(label_filename, ind, DATA_DIR):
    label_filepath = os.path.join(DATA_DIR, label_filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    ind_filepath = os.path.join(DATA_DIR, ind)
    fp = open(ind_filepath, 'r')
    ind_list = [x.strip() for x in fp]
    fp.close()
    ind_np = np.asarray(ind_list, dtype=np.int)
    ind_np = ind_np - 1
    ind_label = label[ind_np, :]
    return paddle.to_tensor(ind_label)


def GenerateCode(model_hash, data_loader, num_data, bit):
    B = np.zeros((num_data, bit), dtype=np.float32)
    for iter, (data_img, _, data_ind) in enumerate(data_loader):
        # data_img = data_img.cuda()
        out = model_hash(data_img)
        B[data_ind.numpy(), :] = paddle.sign(out.cpu()).numpy()
    return B

def PSLDH_Algo(code_length, train):
    batch_size = 70 # 70
    epochs = 225
    learning_rate = 0.002 #0.05
    weight_decay = 10 ** -5
    # model_name = 'ResNet34'
    model_name = 'alexnet'

    alpha = 0.05
    beta = 0.01
    lamda = 0.01 #50
    gamma = 0.2
    sigma = 0.2
    
    ### data processing
    dataset = 'FLICKR-25K'
    if dataset == 'nus':
        data_set = 'nuswide'
        data_name = './label_codes/nus'
    elif dataset == 'coco':
        data_set = 'coco'
        data_name = './label_codes/coco'
    elif dataset == 'FLICKR-25K':
        data_set = 'flickr'
        data_name = './label_codes/flickr'
    file_name = './checkpoint/PSLDH_' + data_set + '_' + str(code_length) + '.pdparams'
    print("*"*10, learning_rate, alpha, beta, lamda, sigma, gamma, code_length, data_set, "*"*10)
    paddle.seed(1)
    dset_database = dp.DatasetProcessing(train=False, database=True, transform=dp.test_transform)
    dset_train = dp.DatasetProcessing(train=True, transform=dp.train_transform)
    dset_test = dp.DatasetProcessing(train=False, database=False, transform=dp.test_transform)
    num_database, num_test = len(dset_database), len(dset_test)

    database_loader = DataLoader(dataset=dset_database, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=True)
    train_loader = DataLoader(dataset=dset_train, 
                              batch_size=batch_size, 
                              num_workers=1,
                              shuffle=True)
    test_loader = DataLoader(dataset=dset_test, 
                             batch_size=batch_size, 
                             num_workers=1,
                             shuffle=False)

    database_labels = dset_database.label
    train_labels = dset_train.label
    test_labels = dset_test.label

    label_size = train_labels.shape
    nclass = label_size[1]

    if os.path.exists(data_name + '_label_code_' + str(code_length) + '.pkl'):
        with open(data_name + '_label_code_' + str(code_length) + '.pkl', 'rb') as f:
            label_code = pickle.load(f)
    else:
        label_model = label_hash_model.Label_net(nclass, code_length)

        optimizer_label = paddle.optimizer.SGD(parameters=label_model.parameters(), learning_rate=0.001, weight_decay=weight_decay)
        scheduler_l = paddle.optimizer.lr.StepDecay(learning_rate=0.001, step_size=100, gamma=0.1, last_epoch=-1)

        labels = paddle.zeros((nclass, nclass)).astype('float32')
        if paddle.is_compiled_with_cuda():
            labels = labels.cuda()
        for i in range(nclass):
            labels[i, i] = 1

        one_hot = paddle.ones((1, nclass)).astype('float32')
        I = paddle.eye(nclass).astype('float32')
        if paddle.is_compiled_with_cuda():
            one_hot = one_hot.cuda()
            I = I.cuda()
        relu = nn.ReLU()
        for i in range(200):
            scheduler_l.step()
            code = label_model(labels)
            loss1 = relu((paddle.matmul(code, code.transpose((1, 0))) - code_length * I))
            loss1 = loss1.pow(2).sum() / (nclass * nclass)
            loss_b = paddle.matmul(one_hot, code).pow(2).sum() / nclass
            re = (paddle.sign(code) - code).pow(2).sum() / nclass
            loss = loss1 + alpha * loss_b + beta * re
            optimizer_label.clear_grad()
            loss.backward()
            optimizer_label.step()
        label_model.eval()
        code = label_model(labels)
        label_code = paddle.sign(code)
        with open(data_name + '_label_code_' + str(code_length) + '.pkl', 'wb') as f:
            pickle.dump(label_code.numpy(), f)

    label_code = paddle.to_tensor(label_code)
    hash_model = image_hash_model.HASH_Net(model_name, code_length)
    optimizer_hash = paddle.optimizer.SGD(parameters=hash_model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate, step_size=300, gamma=0.3, last_epoch=-1)
    ## training
    if train:
        print('Training...')
        for epoch in range(epochs):
            scheduler.step()
            epoch_loss = 0.0
            epoch_loss_r = 0.0
            epoch_loss_e = 0.0
            ## training epoch
            for iter, (train_img, train_label, batch_ind) in enumerate(train_loader):
                train_img = paddle.to_tensor(train_img)
                train_label = paddle.to_tensor(train_label)
                optimizer_hash.clear_grad()
                train_label = paddle.squeeze(train_label)
                if paddle.is_compiled_with_cuda():
                    train_img = train_img.cuda()
                    train_label = train_label.astype('float32').cuda()
                the_batch = len(batch_ind)
                hash_out = hash_model(train_img)
                logit = paddle.matmul(hash_out, paddle.transpose(label_code, [1, 0]))
                max_item = paddle.max(logit, axis=1)[0]
                logit = logit - paddle.reshape(max_item, [-1, 1])
                our_logit = paddle.exp((logit - sigma * code_length) * gamma) * train_label
                mu_logit = (paddle.exp(logit * gamma) * (1 - train_label)).sum(axis=1).reshape([-1, 1]).expand((the_batch, train_label.shape[1])) + our_logit
                loss = - ((paddle.log(our_logit / mu_logit + 1e-5) * train_label).sum(axis=1) / train_label.sum(axis=1)).sum()

                Bbatch = paddle.sign(hash_out)
                regterm = (Bbatch - hash_out).pow(2).sum()
                loss_all = loss / the_batch + regterm * lamda / the_batch

                loss_all.backward()
                optimizer_hash.step()
                epoch_loss += loss_all.item()
                epoch_loss_e += loss.item() / the_batch
                epoch_loss_r += regterm.item() / the_batch
            print('[Train Phase][Epoch: %3d/%3d][Loss_i: %3.5f, Loss_e: %3.5f, Loss_r: %3.5f]' %
                (epoch + 1, epochs, epoch_loss / len(train_loader), epoch_loss_e / len(train_loader),
                epoch_loss_r / len(train_loader)))
        paddle.save(hash_model.state_dict(), file_name)
    else:
        hash_model.set_state_dict(paddle.load(file_name))
        
    ## testing
    print('Testing...')
    hash_model.eval()
    qi = GenerateCode(hash_model, test_loader, num_test, bit)
    ri = GenerateCode(hash_model, database_loader, num_database, bit)
    map = calc_hr.calc_map(qi, ri, test_labels, database_labels)
    print('map:', map)

if __name__=="__main__":
    bit = 16
    train = False
    PSLDH_Algo(bit, train)
