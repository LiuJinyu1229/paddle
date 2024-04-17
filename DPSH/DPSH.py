import paddle
from paddle import nn, optimizer
from paddle.autograd import PyLayer
from paddle.io import DataLoader
from paddle.vision import models, transforms
import os
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm

import utils.DataProcessing as dp
import utils.CalcHammingRanking as CalcHR

import CNN_model

paddle.device.set_device('gpu:6')

def EncodingOnehot(target, nclasses):
    target = paddle.to_tensor(np.array(target))
    target_onehot = paddle.zeros([target.shape[0], nclasses])
    indices = paddle.unsqueeze(target.flatten().astype('int32'), axis=1)
    updates = paddle.ones_like(indices, dtype='float32')
    target_onehot = paddle.scatter(target_onehot, indices, updates)
    return target_onehot

def CalcSim(batch_label, train_label):
    S = (paddle.matmul(batch_label, train_label.t()) > 0).astype('float32')
    return S

def CreateModel(model_name, bit):
    cnn_model = CNN_model.CNNNet(model_name, bit)
    return cnn_model

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    optimizer.set_lr(lr)
    return optimizer

def GenerateCode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = paddle.to_tensor(data_input)
        output = model(data_input)
        B[data_ind.numpy(), :] = paddle.sign(output).cpu().numpy()
    return B

def Logtrick(x):
    lt = paddle.log(1+paddle.exp(-paddle.abs(x))) + paddle.maximum(x, paddle.to_tensor([0.]).cuda())
    return lt

def Totloss(U, B, Sim, lamda, num_train):
    theta = paddle.matmul(U, U.t()) / 2
    t1 = paddle.sum(theta*theta) / (num_train * num_train)
    l1 = (- theta * Sim + Logtrick(theta).numpy()).sum()
    l2 = paddle.sum((U - B)**2)
    l = l1 + lamda * l2
    return l, l1, l2, t1

def DPSH_algo(bit, param):
    # parameters setting
    batch_size = 128
    epochs = 150
    learning_rate = 0.0005
    weight_decay = 10 ** -5
    model_name = 'alexnet'
    data_set = 'flickr'
    nclasses = 24

    filename = param['filename']
    checkpoint_path = './checkpoint/DPSH_' + data_set + '_' + str(bit) + '.pdparams'

    lamda = param['lambda']
    param['bit'] = bit
    param['epochs'] = epochs
    param['learning rate'] = learning_rate
    param['model'] = model_name

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
                             shuffle=False,
                             num_workers=4
                             )
    train_labels = dset_train.label
    test_labels = dset_test.label
    database_labels = dset_database.label
    ### create model
    model = CreateModel(model_name, bit)
    optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)

    ### training phase
    # parameters setting
    B = paddle.zeros([num_train, bit])
    U = paddle.zeros([num_train, bit])
    train_labels_onehot = EncodingOnehot(train_labels, nclasses)
    test_labels_onehot = EncodingOnehot(test_labels, nclasses)

    train_loss = []
    map_record = []

    totloss_record = []
    totl1_record = []
    totl2_record = []
    t1_record = []

    Sim = CalcSim(train_labels_onehot, train_labels_onehot)

    if param['train']:
        print('start training')
        for epoch in range(epochs):
            epoch_loss = 0.0
            ## training epoch
            train_loader = tqdm(train_loader)
            for iter, traindata in enumerate(train_loader, 0):
                train_input, train_label, batch_ind = traindata
                train_label = paddle.squeeze(train_label)
                train_label_onehot = EncodingOnehot(train_label, nclasses)
                train_input, train_label = paddle.to_tensor(train_input), paddle.to_tensor(train_label)
                S = CalcSim(train_label_onehot, train_labels_onehot)
                
                train_outputs = model(train_input)
                for i, ind in enumerate(batch_ind):
                    U[ind, :] = train_outputs[i].detach()
                    B[ind, :] = paddle.sign(train_outputs[i])

                Bbatch = paddle.sign(train_outputs)
                theta_x = paddle.matmul(train_outputs, U.t()) / 2
                logloss = (S*theta_x - Logtrick(theta_x)).sum() / (num_train * len(train_label))
                regterm = paddle.pow(Bbatch - train_outputs, 2).sum() / (num_train * len(train_label))
                
                loss = lamda * regterm - logloss
                epoch_loss += loss.detach().numpy()
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()

            print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (epoch+1, epochs, epoch_loss / len(train_loader)), end='')
            optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)

            l, l1, l2, t1 = Totloss(U, B, Sim, lamda, num_train)
            totloss_record.append(l)
            totl1_record.append(l1)
            totl2_record.append(l2)
            t1_record.append(t1)

            print('[Total Loss: %10.5f][total L1: %10.5f][total L2: %10.5f][norm theta: %3.5f]\n' % (l, l1, l2, t1), end='')

            ### testing during epoch
            qB = GenerateCode(model, test_loader, num_test, bit)
            tB = paddle.sign(B).numpy()
            map_ = CalcHR.CalcMap(qB, tB, np.array(test_labels), np.array(train_labels))
            train_loss.append(epoch_loss / len(train_loader))
            map_record.append(map_)

            print('[Test Phase ][Epoch: %3d/%3d] MAP(retrieval train): %3.5f' % (epoch+1, epochs, map_))
        print('Finished Training')
        paddle.save(model.state_dict(), checkpoint_path)
    else:
        print('load model')
        model.set_state_dict(paddle.load(checkpoint_path))
        print('load model successfully')
    ### evaluation phase
    ## create binary code
    model.eval()
    database_labels_onehot = EncodingOnehot(database_labels, nclasses)
    qB = GenerateCode(model, test_loader, num_test, bit)
    dB = GenerateCode(model, database_loader, num_database, bit)

    map = CalcHR.CalcMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy())
    print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)

    result = {}
    result['qB'] = qB
    result['dB'] = dB
    result['train loss'] = train_loss
    result['map record'] = map_record
    result['map'] = map
    result['param'] = param
    result['total loss'] = totloss_record
    result['l1 loss'] = totl1_record
    result['l2 loss'] = totl2_record
    result['norm theta'] = t1_record
    result['filename'] = filename

    return result

if __name__=='__main__':
    bit = 32
    lamda = 50
    filename = 'log/' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
    param = {}
    param['lambda'] = lamda
    param['filename'] = filename
    param['train'] = True
    result = DPSH_algo(bit, param)
