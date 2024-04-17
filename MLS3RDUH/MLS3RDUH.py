import pickle
import os
import argparse
import logging
import paddle
import time
import scipy
import numpy as np
import paddle.optimizer as optim
import paddle.vision.transforms as transforms

from datetime import datetime
from paddle.io import DataLoader

import utils.data_processing as dp
import utils.hash_model as image_hash_model
import utils.calc_hr as calc_hr
import paddle.nn as nn
import copy
import random
from tqdm import tqdm

paddle.device.set_device('gpu:6')

def GenerateCode(model_hash, data_loader, num_data, bit, k=0):
    B = np.zeros((num_data, bit), dtype='float32')
    for iter, data in enumerate(data_loader, 0):
        data_img, _, data_ind = data
        data_img = paddle.to_tensor(data_img, dtype='float32')
        if k == 0:
            _, out = model_hash(data_img)
            B[data_ind.numpy(), :] = paddle.sign(out).cpu().numpy()
    return B

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def Logtrick(x):
    lt = paddle.log(1+paddle.exp(-paddle.abs(x))) + paddle.maximum(x, paddle.to_tensor([0.], dtype='float32'))
    return lt

def MLS3RDUH_algo(code_length, train):
    random.seed(10)
    batch_size = 128
    epochs = 150
    learning_rate = 0.04 #0.05
    weight_decay = 10 ** -5
    net = './feature/alexnet_flickr'
    model_name = 'alexnet'
    bit = code_length
    lamda = 0.001 #50cp

    data_set = 'flickr'
    file_name = './checkpoint/MLS3RDUH_' + data_set + '_' + str(code_length) + '.pdparams'
    print("*"*10, learning_rate, lamda, code_length, "*"*10)
    ### data processing

    dset_database = dp.DatasetProcessing(train=False, database=True, transform=dp.test_transform)
    dset_train = dp.DatasetProcessing(train=True, transform=dp.train_transform)
    dset_test = dp.DatasetProcessing(train=False, database=False, transform=dp.test_transform)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)
    database_loader = DataLoader(dset_database,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4
                                )
    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    test_loader = DataLoader(dset_test,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4
                             )
    database_labels = dset_database.label
    train_labels = dset_train.label
    test_labels = dset_test.label
    label_size = test_labels.shape
    nclass = label_size[1]

    nnk = int(train_labels.shape[0] * 0.06)
    nno = int(train_labels.shape[0] * 0.06 * 1.5)

    hash_model = image_hash_model.HASH_Net(model_name, code_length)
    optimizer_hash = optim.SGD(parameters=hash_model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=learning_rate, step_size=500, gamma=0.5, last_epoch=-1)

    if os.path.exists(net + '_features4096_labels.pkl'):
        with open(net + '_features4096_labels.pkl', 'rb') as f:
            feature_label = pickle.load(f)
    else:
        x_train = paddle.zeros([num_train, 4096])
        label = paddle.zeros([num_train, nclass])
        train_loader = tqdm(train_loader, desc="Training:")
        for iter, traindata in enumerate(train_loader):
            train_img, train_label, batch_ind = traindata
            train_img = paddle.to_tensor(train_img, dtype='float32')
            train_label = paddle.to_tensor(train_label, dtype='float32')
            feature_out, _ = hash_model(train_img)
            x_train[batch_ind, :] = feature_out.cpu()
            label[batch_ind, :] = paddle.cast(train_label, 'float32')

        feature_label = {'img_feature': x_train.numpy(), 'label': label.numpy()}
        with open(net + '_features4096_labels.pkl', 'wb') as f:
            pickle.dump(feature_label, f)

    feature_x = feature_label['img_feature']
    feature_x = paddle.to_tensor(feature_x, dtype='float32')
    dim = feature_x.shape[0]
    # import ipdb
    # ipdb.set_trace()
    normal = paddle.sqrt((feature_x.pow(2)).sum(1)).reshape([-1, 1])
    normal_feature = feature_x / (normal.expand([dim, 4096]))
    sim1 = paddle.matmul(normal_feature, normal_feature.t())
    final_sim = sim1 * 1.0
    sim = sim1 - paddle.eye(dim).astype('float32')
    top = paddle.uniform([1, dim], min=0, max=1).astype('float32')
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = paddle.argsort(top, axis=1)[0]
        zero = paddle.zeros([dim]).astype('float32')
        zero[top20[-nnk:]] = 1.0
        sim[i, :] = top[0, :] * zero

    A = (sim > 0.0001).astype('float32')
    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)
    aa = dim - (sum_row > 0).astype('float32').sum()
    kk = paddle.argsort(sum_row)
    res_ind = list(range(dim))
    for ind in range(aa):
        res_ind.remove(kk[ind])
    res_ind = random.sample(res_ind, int((dim - aa).item()))
    ind_to_new_id = {}
    for i in range(dim - aa):
        ind_to_new_id[i] = res_ind[i]
    res_ind = paddle.to_tensor(np.asarray(res_ind)).astype('int64')
    # res_ind = res_ind.flatten()
    selected_rows = []
    for i in res_ind:
        selected_rows.append(sim[i, :])
    sim = paddle.stack(selected_rows)
    # sim = sim[res_ind, :]
    # sim = sim[:, res_ind]
    selected_rows = []
    for i in res_ind:
        selected_rows.append(sim[:, i])
    sim = paddle.stack(selected_rows)
    # sim = sim[res_ind]
    sim20 = {}
    dim = int(dim - aa)
    top = paddle.rand([1, dim]).astype('float32')
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = paddle.argsort(top, axis=-1)[0]
        zero = paddle.zeros(dim).astype('float32')
        zero[top20[-nnk:]] = 1.0
        k = list(top20[-nnk:])
        sim20[i] = k
        sim[i, :] = top[0, :] * zero
    A = (sim > 0.0001).astype('float32')

    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)

    sum_row = sum_row.pow(-0.5)
    sim = paddle.diag(sum_row)
    A = paddle.mm(A, sim)
    A = paddle.mm(sim, A)
    alpha = 0.99
    dim = int(dim)
    manifold_sim = (1 - alpha) * paddle.inverse(paddle.eye(dim).astype('float32') - alpha * A)

    manifold20 = {}
    for i in range(dim):
        top[0, :] = manifold_sim[i, :]
        top20 = paddle.argsort(top, axis=-1)[0]
        k = list(top20[-nno:])
        manifold20[i] = k
    print('start calculate final_sim')
    for i in tqdm(range(len(sim20))):
        aa = len(manifold20[i])
        zz = copy.deepcopy(manifold20[i])
        ddd = []
        for k in range(aa):
            if zz[k] in sim20[i]:
                sim20[i].remove(zz[k])
                manifold20[i].remove(zz[k])
                key = zz[k].numpy().item()
                ddd.append(ind_to_new_id[key])
        j = ind_to_new_id[i]
        for l in ddd:
            final_sim[j, l] = 1.0
        for l in sim20[i]:
            key = l.numpy().item()
            final_sim[j, ind_to_new_id[key]] = 0.0
    np.save('final_sim.npy', final_sim)
    print('end calculate final_sim')
    # final_sim = np.load('final_sim.npy')
    f1 = (final_sim > 0.999).astype('float32')
    f1 = ((f1 + f1.T) > 0.999).astype('float32')
    f2 = (final_sim < 0.0001).astype('float32')
    f2 = ((f2 + f2.T) > 0.999).astype('float32')
    final_sim = final_sim * (1. - f2)
    final_sim = final_sim * (1. - f1) + f1
    final_sim = 2 * final_sim - 1.0

    ## training
    if train:
        print("start training...")
        for epoch in range(epochs):
            scheduler.step()
            epoch_loss = 0.0
            epoch_loss_r = 0.0
            epoch_loss_e = 0.0
            train_loader = tqdm(train_loader, desc="Epoch [" + str(epoch) + "]==>Training:")
            for iter, (train_img, train_label, batch_ind) in enumerate(train_loader):
                train_img = paddle.to_tensor(train_img, dtype='float32')
                S = final_sim[batch_ind, :]
                S = S[:, batch_ind]
                the_batch = len(batch_ind)
                _, hash_out = hash_model(train_img)
                loss_all = (paddle.log(paddle.cosh((paddle.matmul(hash_out, hash_out.t()) / float(code_length) \
                                                - paddle.to_tensor(S, dtype='float32'))))).sum() / (the_batch * the_batch)
                Bbatch = paddle.sign(hash_out)
                regterm = paddle.pow((Bbatch - hash_out), 2).sum() / (the_batch * the_batch)
                optimizer_hash.clear_gradients()
                loss_all.backward()
                optimizer_hash.step()
                epoch_loss += loss_all.numpy()
                epoch_loss_r += regterm.numpy()
            print('[Train Phase][Epoch: %3d/%3d][Loss_i: %3.5f, Loss_e: %3.5f, Loss_r: %3.5f]' %
                (epoch + 1, epochs, epoch_loss / len(train_loader), epoch_loss_e / len(train_loader),
                epoch_loss_r / len(train_loader)))

        paddle.save(hash_model.state_dict(), file_name)
        print("end training...")
    else:
        print('---loading trained model---')
        hash_model.set_state_dict(paddle.load(file_name))
        print('---finish loading---')

    ## testing
    print("testing...")
    hash_model.eval()
    qi = GenerateCode(hash_model, test_loader, num_test, bit)
    ri = GenerateCode(hash_model, database_loader, num_database, bit)

    map = calc_hr.calc_map(qi, ri, paddle.to_tensor(np.array(test_labels)).numpy(), paddle.to_tensor(np.array(database_labels)).numpy())
    print('map:', map)

if __name__ == "__main__":
    bit = 64
    train = False     # whether or not to train the model
    MLS3RDUH_algo(bit, train)
