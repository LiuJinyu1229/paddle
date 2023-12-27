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

paddle.device.set_device('gpu:7')

def GenerateCode(model_hash, model_text, data_loader, num_data, bit, k=0):
    B = np.zeros((num_data, bit), dtype=np.float32)
    Bt = np.zeros((num_data, bit), dtype=np.float32)
    kk = 1
    for iter, data in enumerate(data_loader, 0):
        data_img, text, train_label,  data_ind = data
        data_img = paddle.to_tensor(data_img).cuda()
        train_label = paddle.squeeze(train_label)
        train_label = paddle.to_tensor(train_label.astype('float32')).cuda()
        text = paddle.to_tensor(text.astype('float32')).cuda()
        if k == 0:
            out = model_hash(data_img, train_label)
            outt = model_text(text, train_label)
            if kk:
                # print(out.item())
                kk = 0
            B[data_ind.numpy(), :] = paddle.sign(out.numpy())
            Bt[data_ind.numpy(), :] = paddle.sign(outt.numpy())
    return B, Bt

def DAPH_proxy_code_Algo(opt):
    code_length = opt.bit
    bit = opt.bit
    data_set = opt.dataset

    batch_size = 90
    epochs = 80
    learning_rate = 0.0002 #0.05
    weight_decay = 10 ** -5
    model_name = 'alexnet'

    gamma = opt.gamma
    sigma = opt.sigma

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_dta = h5py.File(opt.data_path)
    text_len = np.asarray(all_dta['test_y']).shape[1]
    dset_train = dp.DatasetPorcessing_h5(
        np.asarray(all_dta['train_data']), np.asarray(all_dta['train_y']), np.asarray(all_dta['train_L']),
        transformations)

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    train_labels = dset_train.labels
    label_size = train_labels.shape
    nclass = label_size[1]

    hash_text_model = hash_models.Label_text_Net(text_len, nclass, code_length)
    optimizer_hash_text = paddle.optimizer.SGD(parameters=hash_text_model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler_text = paddle.optimizer.lr.StepDecay(learning_rate, step_size=100, gamma=0.3, last_epoch=-1)

    hash_model = hash_models.Label_Net(model_name, nclass, code_length)
    optimizer_hash = paddle.optimizer.SGD(parameters=hash_model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate, step_size=100, gamma=0.3, last_epoch=-1)

    label_model = hash_models.LabelNet(nclass, code_length)
    optimizer_label = paddle.optimizer.SGD(parameters=label_model.parameters(), learning_rate=0.001, weight_decay=weight_decay)
    scheduler_l = paddle.optimizer.lr.StepDecay(0.001, step_size=100, gamma=0.1, last_epoch=-1)

    labels_cate = paddle.eye(nclass).astype('float32')
    I = paddle.eye(nclass).astype('float32')
    relu = nn.ReLU()
    eps=1e-5
    for epoch in range(epochs):
        label_model.train()
        hash_model.train()
        hash_text_model.train()
        scheduler_l.step()
        scheduler_text.step()
        scheduler.step()
        epoch_loss = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_img, train_text, train_label, batch_ind = traindata
            train_label = paddle.squeeze(train_label)
            train_img = paddle.to_tensor(train_img)
            train_label = paddle.to_tensor(train_label.astype('float32'))
            train_text = paddle.to_tensor(train_text.astype('float32'))
            the_batch = len(batch_ind)
            hash_out = hash_model(train_img, train_label)
            text_out = hash_text_model(train_text, train_label)
            label_out = label_model(labels_cate)

            loss1 = paddle.square(paddle.nn.functional.relu(label_out.matmul(label_out.transpose((1, 0))) - I)).sum()

            logit = hash_out.matmul(label_out.transpose((1, 0)))

            our_logit = paddle.exp((logit - sigma) * gamma) * train_label
            mu_logit = paddle.exp(logit * (1 - train_label) * gamma).sum(1).reshape((-1, 1)).expand((the_batch, train_label.shape[1])) + our_logit
            loss = - ((paddle.log(our_logit / (mu_logit + eps) + eps + 1 - train_label)).sum(1) / train_label.sum(1)).sum()

            logit_text = text_out.matmul(label_out.transpose((1, 0)))

            our_logit_text = paddle.exp((logit_text - sigma) * gamma) * train_label
            mu_logit_text = paddle.exp(logit_text * (1 - train_label) * gamma).sum(1).reshape((-1, 1)).expand((the_batch, train_label.shape[1])) + our_logit_text
            loss_text = - ((paddle.log(our_logit_text / (mu_logit_text + eps) + eps + 1 - train_label)).sum(1) / train_label.sum(1)).sum()

            loss_b = paddle.square(label_out.sum(0)).sum() + paddle.square(text_out.sum(0)).sum() + paddle.square(hash_out.sum(0)).sum()

            B = paddle.nn.functional.normalize(paddle.sign(hash_out))
            reg1 = paddle.square((B * hash_out).sum(1) - 1.).sum()
            Bt = paddle.nn.functional.normalize(paddle.sign(text_out))
            regt = paddle.square((Bt * text_out).sum(1) - 1.).sum()
            B1 = paddle.nn.functional.normalize(paddle.sign(label_out))
            reg2 = paddle.square((B1 * label_out).sum(1) - 1.).sum()
            regterm = reg1 + reg2 + regt

            loss_all = loss + loss_text + regterm * opt.beta + loss1 + loss_b * opt.alpha

            optimizer_hash.clear_gradients()
            optimizer_label.clear_gradients()
            optimizer_hash_text.clear_gradients()
            loss_all.backward()
            optimizer_hash_text.step()
            optimizer_hash.step()
            optimizer_label.step()
            epoch_loss += loss_all.item() / the_batch
        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' %
              (epoch + 1, epochs, epoch_loss / len(train_loader)))
    cate_input = paddle.eye(train_labels.shape[1]).astype('float32').cuda()
    cate_code = label_model(cate_input).detach()
    anchor_code = paddle.zeros((train_labels.shape[0], code_length)).astype('float32').cuda()
    anchor_codet = paddle.zeros((train_labels.shape[0], code_length)).astype('float32').cuda()
    anchor_label = paddle.zeros((train_labels.shape[0], train_labels.shape[1])).astype('float32').cuda()
    for idx, (img, text, labelst, index) in enumerate(train_loader):
        img = paddle.to_tensor(img).cuda()
        labelst = paddle.to_tensor(labelst.astype('float32')).cuda()
        text = paddle.to_tensor(text.astype('float32')).cuda()
        with paddle.no_grad():
            code = hash_model(img, labelst)
            codet = hash_text_model(text, labelst)
            index = paddle.to_tensor(index, dtype='int32')
            anchor_code = paddle.scatter(anchor_code, index, code.detach())
            anchor_codet = paddle.scatter(anchor_codet, index, codet.detach())
            anchor_label = paddle.scatter(anchor_label, index, labelst)

    with open('./data_aware_proxy_codes/' + data_set + '_' + str(code_length) + '.pkl', 'wb') as f:
        pickle.dump({'image_code': anchor_code.numpy(), 'text_code': anchor_codet.numpy(), 'label': anchor_label.numpy(), 'class_code': cate_code.numpy()}, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.2, type=float, help='alpha')
    parser.add_argument('--beta', default=0.2, type=float, help='beta')
    parser.add_argument('--gamma', default=20, type=int, help='gamma')
    parser.add_argument('--sigma', default=0.2, type=float, help='sigma')
    parser.add_argument('--bit', default=64, type=int, help='hash code length')
    parser.add_argument('--dataset', default='iaprtc', type=str, help='dataset name')
    parser.add_argument('--data_path', default='/home1/ljy/dataset/iaprtc.h5', type=str, help='dataset path')
    opt = parser.parse_args()
    DAPH_proxy_code_Algo(opt)
