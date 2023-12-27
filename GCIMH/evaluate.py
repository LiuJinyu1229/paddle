from tqdm import tqdm
import paddle

def mean_average_precision(query_code,
                           retrieval_code,
                           query_targets,
                           retrieval_targets,
                           device,
                           topk=None
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (paddle.Tensor): Query data hash code.
        retrieval_code (paddle.Tensor): Retrieval data hash code.
        query_targets (paddle.Tensor): Query data targets, one-hot
        retrieval_targets (paddle.Tensor): retrieval data targets, one-hot
        device (paddle.device): Using CPU or GPU.
        topk: int

    Returns:
        meanAP (float): Mean Average Precision.
    """
    query_code = query_code.cuda(device)
    retrieval_code = retrieval_code.cuda(device)
    query_targets = query_targets.cuda(device)
    retrieval_targets = retrieval_targets.cuda(device)
    num_query = query_targets.shape[0]
    if topk == None:
        topk = retrieval_targets.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).astype('float32')

        # Calculate hamming distance
        hamming_dist = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[paddle.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().astype('int32').item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = paddle.linspace(1, retrieval_cnt, retrieval_cnt).cuda(device)

        # Acquire index
        index = (paddle.nonzero(retrieval).squeeze() + 1.0).astype('float32')

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP.item()

def pr_curve(query_code, retrieval_code, query_targets, retrieval_targets, device):
    """
    P-R curve.
    Args
        query_code(paddle.Tensor): Query hash code.
        retrieval_code(paddle.Tensor): Retrieval hash code.
        query_targets(paddle.Tensor): Query targets.
        retrieval_targets(paddle.Tensor): Retrieval targets.
        device (paddle.device): Using CPU or GPU.
    Returns
        P(paddle.Tensor): Precision.
        R(paddle.Tensor): Recall.
    """
    num_query = query_code.shape[0]
    num_bit = query_code.shape[1]
    P = paddle.zeros([num_query, num_bit + 1], dtype='float32').cuda(device)
    R = paddle.zeros([num_query, num_bit + 1], dtype='float32').cuda(device)
    for i in tqdm(range(num_query)):
        gnd = (query_targets[i].unsqueeze(0).matmul(retrieval_targets.t()) > 0).astype('float32').squeeze()
        tsum = paddle.sum(gnd)
        if tsum == 0:
            continue
        hamm = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())
        tmp = (hamm <= paddle.arange(0, num_bit + 1).reshape([-1, 1]).astype('float32').cuda(device)).astype('float32')
        total = paddle.sum(tmp, axis=-1)
        total = total + (total == 0).astype('float32') * 0.1
        t = gnd * tmp
        count = paddle.sum(t, axis=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).astype('float32').sum(axis=0)
    mask = mask + (mask == 0).astype('float32') * 0.1
    P = paddle.sum(P, axis=0) / mask
    R = paddle.sum(R, axis=0) / mask

    return P, R