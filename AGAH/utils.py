from PIL import Image
import numpy as np
import paddle

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = paddle.unsqueeze(B1, 0)
    distH = 0.5 * (q - paddle.matmul(B1, paddle.transpose(B2, [1, 0])))
    return distH

def calc_map_k(qB, rB, query_L, retrieval_L, k):
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = paddle.unsqueeze(q_L, 0)
        gnd = (paddle.matmul(q_L, paddle.transpose(retrieval_L, [1, 0])) > 0).astype('float32').squeeze()
        tsum = paddle.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = paddle.argsort(hamm)
        ind = paddle.squeeze(ind)
        # gnd = gnd[ind]
        gnd = paddle.gather(gnd, ind, axis=0)
        total = min(k, int(tsum))
        count = paddle.arange(1, total + 1).astype('float32')
        tindex = paddle.nonzero(gnd)[:total].squeeze().astype('float32') + 1.0
        count = paddle.to_tensor(count, dtype='float32')
        map = map + paddle.mean(count / tindex)
    map = map / num_query
    return map

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qB, rB, query_L, retrieval_L):
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = paddle.unsqueeze(q_L, 0)
        gnd = (paddle.matmul(q_L, paddle.transpose(retrieval_L, [1, 0])) > 0).astype('float32').squeeze()
        tsum = paddle.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = paddle.argsort(hamm)
        ind = paddle.squeeze(ind)
        gnd = gnd[ind]
        total = int(tsum)
        count = paddle.arange(1, total + 1).astype('float32')
        tindex = paddle.nonzero(gnd)[:total].squeeze().astype('float32') + 1.0
        count = paddle.to_tensor(count, dtype='float32')
        map = map + paddle.mean(count / tindex)
    map = map / num_query
    return map


def image_from_numpy(x):
    if x.max() > 1.0:
        x = x / 255
    if type(x) != np.ndarray:
        x = x.numpy()
    im = Image.fromarray(np.uint8(x * 255))
    im.show()


def pr_curve(qB, rB, queryL, retrievalL):
    dim = np.shape(rB)
    bit = dim[1]
    all_ = dim[0]
    precision = np.zeros(bit + 1)
    recall = np.zeros(bit + 1)
    num_query = queryL.shape[0]
    num_database = retrievalL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        all_sum = np.sum(gnd).astype(np.float32)
        # print(all_sum)
        if all_sum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB)
        # print(hamm.shape)
        ind = np.argsort(hamm)
        # print(ind.shape)
        gnd = gnd[ind]
        hamm = hamm[ind]
        hamm = hamm.tolist()
        # print(len(hamm), num_database - 1)
        max_ = hamm[num_database - 1]
        max_ = int(max_)
        t = 0
        for i in range(1, max_):
            if i in hamm:
                idd = hamm.index(i)
                if idd != 0:
                    sum1 = np.sum(gnd[:idd])
                    precision[t] += sum1 / idd
                    recall[t] += sum1 / all_sum
                else:
                    precision[t] += 0
                    recall[t] += 0
                t += 1
        # precision[t] += all_sum / num_database
        # recall[t] += 1
        for i in range(t,  bit + 1):
            precision[i] += all_sum / num_database
            recall[i] += 1
    true_recall = recall / num_query
    precision = precision / num_query
    return precision, true_recall


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
