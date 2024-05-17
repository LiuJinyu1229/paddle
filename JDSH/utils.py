import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer
import paddle.vision.datasets as dsets
from paddle.vision import transforms
import paddle.vision as vision
import math
import numpy as np
import logging
import os.path as osp


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def p_topK(qB, rB, queryL, retrievalL, topk=1000):
    n = topk // 100
    precision = np.zeros(n)
    num_query = queryL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hamming_dist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        for i in range(1, n + 1):
            a = gnd[:i * 100]
            precision[i - 1] += float(a.sum()) / (i * 100.)
    a_precision = precision / num_query
    return a_precision


def compress(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, _, _) in enumerate(train_loader):
        var_data_I = paddle.to_tensor(data_I).cuda()
        _,_,code_I = model_I(var_data_I)
        code_I = paddle.sign(code_I)
        re_BI.extend(code_I.cpu())
        
        var_data_T = paddle.to_tensor(data_T, dtype='float32').cuda()
        code_T = model_T(var_data_T)
        code_T = paddle.sign(code_T)
        re_BT.extend(code_T.cpu())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, _, _) in enumerate(test_loader):
        var_data_I = paddle.to_tensor(data_I).cuda()
        _,_,code_I = model_I(var_data_I)
        code_I = paddle.sign(code_I)
        qu_BI.extend(code_I.cpu())
        
        var_data_T = paddle.to_tensor(data_T, dtype='float32').cuda()
        code_T = model_T(var_data_T)
        code_T = paddle.sign(code_T)
        qu_BT.extend(code_T.cpu())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = train_dataset.labels
    re_L = paddle.to_tensor(np.array(re_L)).cuda()

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = test_dataset.labels
    qu_L = paddle.to_tensor(np.array(qu_L)).cuda()
    re_BI = paddle.to_tensor(re_BI).cuda()
    re_BT = paddle.to_tensor(re_BT).cuda()
    qu_BI = paddle.to_tensor(qu_BI).cuda()
    qu_BT = paddle.to_tensor(qu_BT).cuda()
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = paddle.unsqueeze(B1, 0)
    distH = 0.5 * (q - paddle.matmul(B1, paddle.transpose(B2, [1, 0])))
    return distH

def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        q_L = qu_L[iter]
        # if len(q_L.shape) < 2:
        #     q_L = paddle.unsqueeze(q_L, 0)
        gnd = (paddle.matmul(q_L, paddle.transpose(re_L, [1, 0])) > 0).astype('float32').squeeze()
        tsum = paddle.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qu_B[iter, :], re_B)
        ind = paddle.argsort(hamm)
        ind = paddle.squeeze(ind)
        gnd = gnd[ind]
        total = min(topk, int(tsum))
        count = paddle.arange(1, total + 1).astype('float32')
        tindex = paddle.nonzero(gnd)[:total].squeeze().astype('float32') + 1.0
        count = paddle.to_tensor(count, dtype='float32')
        map = map + paddle.mean(count / tindex)
    map = map / num_query
    return map

def gen_adj(A):
    D = paddle.pow(A.sum(1).astype('float32'), -0.5)
    D = paddle.diag(D)
    adj = paddle.matmul(paddle.matmul(A, D).t(), D)
    return adj

def logger():
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = 'log.txt'
    log_dir = './log'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger