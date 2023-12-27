import pickle
import os
import argparse
import logging
import torch
import time
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.utils.data import DataLoader

import utils.data_processing as dp
import utils.hash_model as image_hash_model
import utils.label_hash_model as label_hash_model
import utils.calc_hr as calc_hr
import torch.nn as nn


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
    return torch.from_numpy(ind_label)


def GenerateCode(model_hash, data_loader, num_data, bit):
    B = np.zeros((num_data, bit), dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_img, _, data_ind = data
        data_img = data_img.cuda()
        out = model_hash(data_img)
        B[data_ind.numpy(), :] = torch.sign(out.data.cpu()).numpy()
    return B


def data_loss(label_code, label, train_b, train_label, weighted, indicator, gamma=0.2, sigma=0.3, epsilon=1e-5):
    code_length = label_code.shape[1]
    sigma = sigma * code_length
    logit = train_b.mm(label_code.t())

    max_item = logit.max(1)[0]
    logit = logit - max_item.view(-1, 1)
    sim_label = (train_label.mm(label.t()) > 0).float()
    our_logit = torch.exp((logit - sigma) * gamma) * sim_label
    mu_logit = (torch.exp(logit * gamma) * (1 - sim_label)).sum(1).view(-1, 1) + our_logit
    loss = - (((torch.log(our_logit / (mu_logit + epsilon) + epsilon) * sim_label).sum(1) / (
            sim_label.sum(1) + epsilon)) * weighted).mean()

    logit_code = train_b.mm(train_b.t())
    max_item = logit_code.max(1)[0]
    logit_code = logit_code - max_item.view(-1, 1)
    sims = (train_label.mm(train_label.t()) > 0).float()
    our_logits = torch.exp((logit_code - sigma) * gamma) * sims * indicator.view(1, -1)
    mu_logits = (torch.exp(logit_code * gamma) * (1 - sims) * indicator.view(1, -1)).sum(1).view(-1, 1) + our_logits
    loss_codes = - ((torch.log(our_logits / (mu_logits + epsilon) + epsilon) * sims * indicator.view(1, -1)).sum(1) / ((sims * indicator.view(1, -1)).sum(1) + epsilon)).sum() / (((sims * indicator.view(1, -1)).sum(1) > 0).sum() + epsilon)

    return loss, loss_codes


def PSLDH_Algo(code_length):
    DATA_DIR = '/data/home/trc/mat/imagenet'
    LABEL_FILE = 'label_hot.txt'
    IMAGE_FILE = 'images_name.txt'
    DATABASE_FILE = 'database_ind_ph.txt'
    TRAIN_FILE = 'train_ind_ph.txt'
    TEST_FILE = 'test_ind_ph.txt'
    data_set = 'imagenet_vgg11'

    top_k = 5000

    os.environ['CUDA_VISIBLE_DEVICES'] = str(6)

    batch_size = 70
    epochs = 120
    learning_rate = 0.001  # 0.05
    weight_decay = 10 ** -5
    model_name = 'vgg11'

    alpha = 0.05
    beta = 0.01
    lamda = 0.01  # 50
    gamma = 0.2
    sigma = 0.2
    print("*" * 10, learning_rate, alpha, beta, lamda, sigma, gamma, code_length, top_k, data_set, "*" * 10)
    ### data processing

    dataset = 'MS-COCO'
    if dataset == 'NUS-WIDE':
        data_name = './label_codes/nus'
        data_dir = '/home/trc/datasets/dataset_mat/nus.h5'
    elif dataset == 'MS-COCO':
        data_name = './label_codes/coco'
        data_dir = '/home/trc/datasets/dataset_mat/coco.h5'
    elif dataset == 'FLICKR-25K':
        data_dir = '/home/trc/datasets/dataset_mat/mir25_crossmodal.h5'
    dset_database = dp.DatasetProcessing(data_dir, 'dataset')
    dset_train = dp.DatasetProcessing(data_dir, 'train')
    dset_test = dp.DatasetProcessing(data_dir, 'test')
    num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)

    database_loader = DataLoader(dset_database, batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    database_labels = dset_database.label
    train_labels = dset_train.label
    test_labels = dset_test.label

    label_size = train_labels.size()
    nclass = label_size[1]

    if os.path.exists(data_name + '_label_code_' + str(code_length) + '.pkl'):
        with open(data_name + '_label_code_' + str(code_length) + '.pkl', 'rb') as f:
            label_code = pickle.load(f)
    else:
        label_model = label_hash_model.Label_net(nclass, code_length)
        label_model.cuda()

        optimizer_label = optim.SGD(label_model.parameters(), lr=0.001, weight_decay=weight_decay)
        scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=100, gamma=0.1, last_epoch=-1)

        labels = torch.zeros((nclass, nclass)).type(torch.FloatTensor).cuda()
        for i in range(nclass):
            labels[i, i] = 1

        one_hot = torch.ones((1, nclass)).type(torch.FloatTensor).cuda()
        I = torch.eye(nclass).type(torch.FloatTensor).cuda()
        relu = nn.ReLU()
        for i in range(200):
            scheduler_l.step()
            code = label_model(labels)
            loss1 = relu((code.mm(code.t()) - code_length * I))
            loss1 = loss1.pow(2).sum() / (nclass * nclass)
            loss_b = one_hot.mm(code).pow(2).sum() / nclass
            re = (torch.sign(code) - code).pow(2).sum() / nclass
            loss = loss1 + alpha * loss_b + beta * re
            optimizer_label.zero_grad()
            loss.backward()
            optimizer_label.step()
        label_model.eval()
        code = label_model(labels)
        label_code = torch.sign(code)
        with open(data_name + '_label_code_' + str(code_length) + '.pkl', 'wb') as f:
            pickle.dump(label_code, f)

    hash_model = image_hash_model.HASH_Net(model_name, code_length)
    hash_model.cuda()
    optimizer_hash = optim.SGD(hash_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hash, step_size=100, gamma=0.3, last_epoch=-1)
    B = np.zeros((num_train, bit), dtype=np.float32)
    labels = torch.zeros((nclass, nclass)).type(torch.FloatTensor).cuda()
    for i in range(nclass):
        labels[i, i] = 1
    for epoch in range(epochs):
        yita = (epoch // 10) * 0.1
        hash_model.eval()
        for iter, data in enumerate(train_loader, 0):
            img, label, index = data
            data_img = img.cuda()
            out = hash_model(data_img)
            B[index.numpy(), :] = torch.sign(out.data.cpu()).numpy()

        map_score = calc_hr.calc_map_score(B, B, train_labels.numpy(), train_labels.numpy())
        map_score = torch.tensor(map_score).float().cuda()
        sim_train = (train_labels.mm(train_labels.t()) > 0).float().cuda()
        mean_scores = sim_train.mm(map_score.view(-1, 1)).view(-1) / sim_train.sum(1)
        indicator = (map_score > 0.8) * (map_score > mean_scores) * 1.
        weighted = (map_score > mean_scores) * 1.
        weighted = torch.exp((1 + mean_scores - map_score) * (1 - weighted))
        weighted = weighted.detach()

        scheduler.step()
        epoch_loss = 0.0
        epoch_loss_c = 0.0
        epoch_loss_r = 0.0
        epoch_loss_e = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_img, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)
            train_img = train_img.cuda()
            train_label = train_label.type(torch.FloatTensor).cuda()
            the_batch = len(batch_ind)
            hash_out = hash_model(train_img)
            # logit = hash_out.mm(label_code.t())
            # our_logit = torch.exp((logit - sigma * code_length) * gamma) * train_label
            # mu_logit = (torch.exp(logit * gamma) * (1 - train_label)).sum(1).view(-1, 1).expand(the_batch,
            #                                                                                     train_label.size()[
            #                                                                                         1]) + our_logit
            # loss = - ((torch.log(our_logit / mu_logit + 1 - train_label)).sum(1) / train_label.sum(1)).sum() / the_batch
            loss, loss_code = data_loss(label_code, labels, hash_out, train_label, weighted[batch_ind], indicator[batch_ind], gamma=0.2, sigma=0.2)
            Bbatch = torch.sign(hash_out)
            regterm = (Bbatch - hash_out).pow(2).sum()
            loss_all = loss + regterm * lamda / the_batch + loss_code * yita

            optimizer_hash.zero_grad()
            loss_all.backward()
            optimizer_hash.step()
            epoch_loss += loss_all.item()
            epoch_loss_e += loss.item()
            epoch_loss_c += loss_code.item()
            epoch_loss_r += regterm.item() / the_batch
        print('[Train Phase][Epoch: %3d/%3d][Loss_i: %3.5f, Loss_code: %3.5f, Loss_e: %3.5f, Loss_r: %3.5f]' %
              (epoch + 1, epochs, epoch_loss / len(train_loader), epoch_loss_c / len(train_loader), epoch_loss_e / len(train_loader),
               epoch_loss_r / len(train_loader)))

        if (epoch + 1) % 30 == 0:
            hash_model.eval()
            qi = GenerateCode(hash_model, test_loader, num_test, bit)
            ri = GenerateCode(hash_model, database_loader, num_database, bit)
            map = calc_hr.calc_topMap(qi, ri, test_labels.numpy(), database_labels.numpy(), top_k)
            map_all = calc_hr.calc_map(qi, ri, test_labels.numpy(), database_labels.numpy())
            print('test_map:', map, 'map_all:', map_all)
    '''
    training procedure finishes, evaluation
    '''


if __name__ == "__main__":
    bits = [64, 32, 16]
    for bit in bits:
        PSLDH_Algo(bit)
