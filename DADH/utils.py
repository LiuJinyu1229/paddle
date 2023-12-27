import numpy as np

# def calc_hamming_dist(B1, B2):
#     q = B2.shape[1]
#     if len(B1.shape) < 2:
#         B1 = B1.unsqueeze(0)
#     distH = 0.5 * (q - B1.mm(B2.t()))
#     return distH

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.t()))
    return distH

def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        # gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(queryL[iter, :], retrievalL.t()) > 0).astype(np.float32)
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


# def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
#     num_query = query_label.shape[0]
#     map = 0.
#     if k is None:
#         k = retrieval_label.shape[0]
#     for i in range(num_query):
#         gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
#         tsum = torch.sum(gnd)
#         if tsum == 0:
#             continue
#         hamm = calc_hamming_dist(qB[i, :], rB)
#         _, ind = torch.sort(hamm)
#         ind.squeeze_()
#         gnd = gnd[ind]
#         total = min(k, int(tsum))
#         count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
#         tindex = torch.nonzero(gnd, as_tuple=False)[:total].squeeze().type(torch.float) + 1.0
#         map += torch.mean(count / tindex)
#     map = map / num_query
#     return map
