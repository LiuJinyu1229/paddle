import numpy as np
import pickle

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose(1, 0)))
    return distH


def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose(1, 0)) > 0).astype(np.float32)
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

def calc_HammingMap(qB, rB, qf, rf, queryL, retrievalL, topr):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    topr += 0.5
    num_query = queryL.shape[0]
    print(num_query)
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose(1, 0)) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        hamm_s = hamm[ind]
        gnd_sum = int(np.sum(hamm_s < topr))
        select_ind = ind[0:gnd_sum]
        return_f = rf[select_ind, :]
        i_f = np.tile(qf[iter, :], (gnd_sum, 1))
        ed = ((return_f - i_f)**2).sum(1)
        f_ind = np.argsort(ed)
        select_gnd = gnd[select_ind]
        tgnd = select_gnd[f_ind]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap

def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    print(num_query)
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose(1, 0)) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap
def pr_curve(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    dim = np.shape(rB)
    bit = dim[1]
    all_ = dim[0]
    precision = np.zeros(bit + 1)
    recall = np.zeros(bit + 1)
    num_query = queryL.shape[0]
    num_database = retrievalL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose(1, 0)) > 0).astype(np.float32)
        all_sum = np.sum(gnd).astype(np.float32)
        # print(all_sum)
        if all_sum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
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
    print(true_recall)
    print(precision)
    return true_recall, precision
def CalcNDCG_N(N, qB, rB, queryL, retrievalL):
    num_q = qB.shape[0]
    a_NDCG = 0.0
    NDCG = 0.0
    for i in range(num_q):
        DCG = 0.0
        max_DCG = 0.0
        sim = (np.dot(queryL[i, :], retrievalL.transpose(1, 0))).astype(np.float32)
        # qL = np.sum(queryL[i,:]).astype(np.float32)
        # rL = np.sum(retrievalL, axis=1).astype(np.float32)
        # L = np.power(qL * rL, 0.5).astype(np.float32)
        # sim = sim / L
        hamm = calc_hammingDist(qB[i, :], rB)
        ind = np.argsort(hamm)
        sim_sort = np.argsort(sim)
        for k in range(N):
            gain = 2 ** sim[ind[k]] - 1
            gain_max = 2 ** sim[sim_sort[- k - 1]] - 1
            log = np.log2(k + 2)
            DCG += gain / log
            max_DCG += gain_max / log
        NDCG += DCG / max_DCG
    a_NDCG = NDCG / num_q
    return a_NDCG
