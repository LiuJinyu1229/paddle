import paddle
import math
import numpy as np


def compress_wiki(train_loader, test_loader, modeli, modelt, train_dataset, test_dataset, classes=10):
    re_BI = []
    re_BT = []
    re_L = []
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        var_data_I = paddle.to_tensor(data_I)
        _,_,code_I = modeli(var_data_I)
        code_I = paddle.sign(code_I)
        re_BI.extend(code_I.numpy())
        re_L.extend(target)
        
        var_data_T = paddle.to_tensor(data_T)
        _,_,code_T = modelt(var_data_T)
        code_T = paddle.sign(code_T)
        re_BT.extend(code_T.numpy())
    
    qu_BI = []
    qu_BT = []
    qu_L = []
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        var_data_I = paddle.to_tensor(data_I)
        _,_,code_I = modeli(var_data_I)
        code_I = paddle.sign(code_I)
        qu_BI.extend(code_I.numpy())
        qu_L.extend(target)
        
        var_data_T = paddle.to_tensor(data_T)
        _,_,code_T = modelt(var_data_T)
        code_T = paddle.sign(code_T)
        qu_BT.extend(code_T.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.eye(classes)[np.array(re_L)]

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.eye(classes)[np.array(qu_L)]
    
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def compress(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset):
    re_BI = []
    re_BT = []
    re_L = []
    for _, (data_I, data_T, _, _) in enumerate(train_loader):
        # import ipdb
        # ipdb.set_trace()
        var_data_I = paddle.to_tensor(data_I)
        _,_,code_I = model_I(var_data_I)
        code_I = paddle.sign(code_I)
        re_BI.extend(code_I.numpy())
        
        var_data_T = paddle.to_tensor(data_T)
        _,_,code_T = model_T(var_data_T)
        code_T = paddle.sign(code_T)
        re_BT.extend(code_T.numpy())
    
    qu_BI = []
    qu_BT = []
    qu_L = []
    for _, (data_I, data_T, _, _) in enumerate(test_loader):
        var_data_I = paddle.to_tensor(data_I)
        _,_,code_I = model_I(var_data_I)
        code_I = paddle.sign(code_I)
        qu_BI.extend(code_I.numpy())
        
        var_data_T = paddle.to_tensor(data_T)
        _,_,code_T = model_T(var_data_T)
        code_T = paddle.sign(code_T)
        qu_BT.extend(code_T.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = train_dataset.train_labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = test_dataset.train_labels
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


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
        gnd = np.dot(qu_L[(iter), :], re_L.transpose()) > 0
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[(iter), :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / tindex)
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[(iter), :], re_L.transpose()) > 0).astype(np.float32
            )
        hamm = calculate_hamming(qu_B[(iter), :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / tindex)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
