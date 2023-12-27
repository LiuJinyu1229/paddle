import paddle
import os
import random
import numpy as np


def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)


def zero2eps(x):
    x[x == 0] = 1
    return x


def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, (np.newaxis)])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity / col_sum
    in_affnty = np.transpose(affinity / row_sum)
    return in_affnty, out_affnty


def affinity_tag_multi(label_1, label_2):
    """
    Use label or plabel to create the graph.
    :param tag1:
    :param tag2:
    :return:
    """
    aff = paddle.matmul(label_1, label_2.t())
    affinity_matrix = aff.astype('float32')
    affinity_matrix = 1 / (1 + paddle.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    return affinity_matrix
    # aff = np.matmul(tag1, tag2.T)
    # affinity_matrix = np.float32(aff)
    # affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    # affinity_matrix = 2 * affinity_matrix - 1
    # in_aff, out_aff = normalize(affinity_matrix)
    # return in_aff, out_aff, affinity_matrix


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map

# Refer
def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def get_COO_matrix(label_matrix: np.ndarray, theshold=None) ->np.ndarray:
    instance_num, label_dim = label_matrix.shape
    if theshold is None:
        theshold = instance_num // 20
    adj = label_matrix.T @ label_matrix
    adj[range(label_dim), range(label_dim)] = instance_num
    adj_norm = adj / instance_num
    return adj, adj_norm

# def calc_hammingDist(B1, B2):
#     q = B2.shape[1]
#     distH = 0.5 * (q - np.dot(B1, B2.transpose()))
#     return distH

# def calculate_map(qB, rB, queryL, retrievalL):
#     # qB: {-1,+1}^{mxq}
#     # rB: {-1,+1}^{nxq}
#     # queryL: {0,1}^{mxl}
#     # retrievalL: {0,1}^{nxl}
#     num_query = queryL.shape[0]
#     map = 0
#     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     print(qB.shape)
#     print(rB.shape)
#     print(queryL.shape)
#     print(retrievalL.shape)

#     for iter in range(num_query):
#         gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
#         tsum = int(np.sum(gnd))
#         if tsum == 0:
#             continue
#         hamm = calc_hammingDist(qB[iter, :], rB)
#         ind = np.argsort(hamm)
#         gnd = gnd[ind]
#         count = np.linspace(1, tsum, tsum)

#         tindex = np.asarray(np.where(gnd == 1)) + 1.0
#         map_ = np.mean(count / (tindex))
#         # print(map_)
#         map = map + map_
#     map = map / num_query
#     # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

#     return map
