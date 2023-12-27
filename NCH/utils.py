import paddle
import logging
import os
import random
import time
import math
import numpy as np


def zero2eps(x):
    x[x == 0] = 1
    return x


def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, (np.newaxis)])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity / col_sum
    in_affnty = np.transpose(affinity / row_sum)
    return in_affnty, out_affnty


def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)


def logger():
    """
    '[0;34m%s[0m': blue
    :return:
    """
    logger = logging.getLogger('PAGN')
    logger.setLevel(logging.DEBUG)
    if not os.path.exists('log/'):
        os.mkdir('log/')
    timeStr = time.strftime('[%m-%d]%H:%M:%S', time.localtime())
    txt_log = logging.FileHandler('log/' + timeStr + '.log')
    txt_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s',
        '%m/%d %H:%M:%S')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('\x1b[0;32m%s\x1b[0m' %
        '[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)
    return logger


def log_params(logger, config: dict):
    logger.info('--- Configs List---')
    for k in config.keys():
        logger.info('--- {:<18}:{}'.format(k, config[k]))


def GEN_S_GPU(label_1, label_2):
    aff = paddle.matmul(label_1, label_2.t())
    affinity_matrix = aff.astype(dtype='float32')
    affinity_matrix = 1 / (1 + paddle.exp(x=-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    return affinity_matrix


def int2bool(flag: int):
    """

    :param flag: -1: False // 1: True
    :return:
    """
    return False if flag == -1 else True


def l2norm(X, dim, eps=1e-08):
    """L2-normalize columns of X
    """
    norm = paddle.pow(x=X, y=2).sum(axis=dim, keepdim=True).sqrt() + eps
    X = paddle.divide(x=X, y=paddle.to_tensor(norm))
    return X

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(qB.shape)
    print(rB.shape)
    print(queryL.shape)
    print(retrievalL.shape)

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = int(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map
