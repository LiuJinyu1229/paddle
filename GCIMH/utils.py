import numpy as np
import scipy.io as sio
import paddle

def calculate_hamming_distance(a, b):
    q = a.shape[-1]
    return 0.5 * (q - paddle.matmul(a, b.t()))

def calculate_s(labels1, labels2):
    s = paddle.matmul(labels1, labels2.t())
    return s


def normalize(x):
    l2_norm = np.linalg.norm(x, axis=1)[:, None]
    l2_norm[np.where(l2_norm == 0)] = 1e-6
    l2_norm = paddle.to_tensor(l2_norm)
    x = x/l2_norm
    return x


def zero_mean(x, mean_val=None):
    if mean_val is None:
        mean_val = np.mean(x, axis=0)
    x -= mean_val
    return x, mean_val

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


if __name__ == '__main__':
    pass
