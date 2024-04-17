import paddle

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = paddle.unsqueeze(B1, 0)
    distH = 0.5 * (q - paddle.matmul(B1, paddle.transpose(B2, [1, 0])))
    return distH

def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
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
        total = min(k, int(tsum))
        count = paddle.arange(1, total + 1).astype('float32')
        tindex = paddle.nonzero(gnd)[:total].squeeze().astype('float32') + 1.0
        count = paddle.to_tensor(count, dtype='float32')
        map = map + paddle.mean(count / tindex)
    map = map / num_query
    return map

if __name__ == '__main__':
    qB = paddle.to_tensor([[1, -1, 1, 1],
                           [-1, -1, -1, 1],
                           [1, 1, -1, 1],
                           [1, 1, 1, -1]])
    rB = paddle.to_tensor([[1, -1, 1, -1],
                           [-1, -1, 1, -1],
                           [-1, -1, 1, -1],
                           [1, 1, -1, -1],
                           [-1, 1, -1, -1],
                           [1, 1, -1, 1]])
    query_L = paddle.to_tensor([[0, 1, 0, 0],
                                [1, 1, 0, 0],
                                [1, 0, 0, 1],
                                [0, 1, 0, 1]])
    retrieval_L = paddle.to_tensor([[1, 0, 0, 1],
                                    [1, 1, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 0, 1, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 1, 0]])

    map = calc_map_k(qB, rB, query_L, retrieval_L)
    print(map)