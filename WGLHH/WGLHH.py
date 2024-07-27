import paddle
from paddle.vision import transforms
from paddle.vision import models
import os
import numpy as np
import pickle
from datetime import datetime
import argparse
from paddle.io import DataLoader

import utils.DataProcessing as dp
import utils.calc_hr as CalcHR
import time
import utils.CNN_model as CNN_model

paddle.device.set_device('gpu:6')

def CalcSim(batch_label, train_label):
    S = (paddle.matmul(batch_label, train_label.t()) > 0).astype('float32')
    return S

def GenerateCode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    f = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader):
        data_input, _, data_ind = data
        data_input = paddle.to_tensor(data_input).cuda()
        _, output = model(data_input)
        output_cpu = output.cpu()
        B[data_ind.numpy(), :] = paddle.sign(output_cpu).numpy()
        f[data_ind.numpy(), :] = output_cpu.numpy()
    return B, f

def WGLHH_algo(bit, train):
    # parameters setting
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    batch_size = 50
    epochs = 500
    learning_rate = 0.005
    weight_decay = 10 ** -4
    model_name = 'alexnet'
    theta = 2.0
    gamma = 0.001
    alpha = 0.1
    containt0 = 0.0001
    data_set = 'flickr'
    file_name = './checkpoint/WGLHH_' + data_set + '_' + str(bit) + '.pdparams'

    ## MS COCO
    if bit < 32:
        kt = 0.1
    elif bit == 64:
        kt = 0.7
    else:
        kt = 0.6

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
    database_labels_onehot = dset_database.label
    test_labels_onehot = dset_test.label

    ### create model
    model = CNN_model.cnn_model(model_name, bit)
    optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate, step_size=401, gamma=0.5, last_epoch=-1, verbose=False)
    ## Training
    if train:
        print('---start training---')
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_lossq = 0.0
            scheduler.step()
            ## training epoch
            for iter, traindata in enumerate(train_loader):
                train_input, train_label, batch_ind = traindata
                train_label = paddle.squeeze(train_label)

                train_label_onehot = train_label.astype('float32')
                train_input, train_label = paddle.to_tensor(train_input).cuda(), paddle.to_tensor(train_label).cuda()
                S = CalcSim(train_label_onehot, train_label_onehot)

                model.clear_gradients()
                _, train_outputs = model(train_input)

                normal1 = paddle.sqrt((train_outputs**2).sum(1)).reshape([-1, 1])
                normal_code = train_outputs / (normal1.expand([train_label.shape[0], bit]))
                c_sim = paddle.matmul(normal_code, paddle.transpose(normal_code, [1, 0]))
                weight_s = paddle.exp(paddle.abs((0.5 * c_sim + 0.5 - S)))
                dh = 0.5 * float(bit) * (1 - c_sim)

                lq = (paddle.sign(train_outputs) - train_outputs)**2

                pq = paddle.exp(-alpha * (dh ** 2))
                lpq = pq * paddle.log((theta * pq + containt0) / ((theta - 1) * pq + S + containt0)) + \
                    S * paddle.log((theta * S + (1 - S) * 0.1) / (
                        (theta - 1) * S + (1 - S) * 0.0001 + pq))
                loss = ((S * kt + 1) * weight_s * lpq).sum() / (
                            train_label.shape[0] * train_label.shape[0]) + gamma * lq.sum() / train_label.shape[0]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_lossq += (lq.sum() / train_label.shape[0]).numpy().item()
            print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f][Lossq1: %3.5f]' %
                        (epoch + 1, epochs, epoch_loss / len(train_loader), epoch_lossq / len(train_loader)))
        paddle.save(model.state_dict(), file_name)
        print('---finish training---')
    else:
        print('---loading trained model---')
        model.set_state_dict(paddle.load(file_name))
        print('---finish loading---')
    ## Testing
    print('---start testing---')
    model.eval()
    qB, qf = GenerateCode(model, test_loader, num_test, bit)
    dB, df = GenerateCode(model, database_loader, num_database, bit)
    map_n = CalcHR.calc_HammingMap(qB, dB, qf, df, test_labels_onehot, database_labels_onehot, 2)
    print('map@2:', map_n)
    print('---finish testing---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bit', type=int, default=16)
    parser.add_argument('--train', action='store_true', help='Training mode')
    args = parser.parse_args()
    WGLHH_algo(args.bit, args.train)
