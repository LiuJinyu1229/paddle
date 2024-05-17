import paddle
import paddle.optimizer as optim
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from model_loader import load_model
from evaluate import mean_average_precision
import paddle.nn as nn

def train(logger,
          train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          multi_labels,
          code_length,
          num_features,
          alpha,
          beta,
          max_iter,
          arch,
          lr,
          verbose,
          evaluate_interval,
          snapshot_interval,
          topk,
          checkpoint,
          test_label,
          train_label,
          database_label
          ):
    """
    Training model.

    Args
        train_dataloader(paddle.io.DataLoader): Training data loader.
        query_dataloader(paddle.io.DataLoader): Query data loader.
        retrieval_dataloader(paddle.io.DataLoader): Retrieval data loader.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        num_features(int): Number of features.
        alpha, beta(float): Hyper-parameters.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        lr(float): Learning rate.
        verbose(bool): Print log.
        evaluate_interval(int): Interval of evaluation.
        snapshot_interval(int): Interval of snapshot.
        topk(int): Calculate top k data points map.
        checkpoint(str, optional): Path of checkpoint.

    Returns
        None
    """
    # Model, optimizer, criterion
    model = load_model(arch, code_length)
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9, weight_decay=1e-5)
    criterion = SSDH_Loss()

    # Extract features
    logger.info("Begin to extract features")
    features = extract_features(model, train_dataloader, num_features, verbose)
    logger.info("extract features successfully")

    # Generate similarity matrix
    S = generate_similarity_matrix(features, alpha, beta).cuda()

    # Training
    model.train()
    for epoch in range(max_iter):
        n_batch = len(train_dataloader)
        for i, (data, _, index) in enumerate(train_dataloader):
            # Current iteration
            cur_iter = epoch * n_batch + i + 1

            data = paddle.to_tensor(data)
            optimizer.clear_grad()

            v = model(data)
            H = paddle.matmul(v, v.t()) / code_length
            targets = S[index, :][:, index]
            loss = criterion(H, targets)

            loss.backward()
            optimizer.step()

            # Print log
            if verbose:
                logger.info('[epoch:{}][Batch:{}/{}][loss:{:.4f}]'.format(epoch+1, i+1, n_batch, loss.numpy().item()))

            # Evaluate
            if cur_iter % evaluate_interval == 0:
                mAP = evaluate(logger,
                               model,
                               query_dataloader,
                               retrieval_dataloader,
                               code_length,
                               topk,
                               multi_labels,
                               test_label,
                               train_label,
                               database_label
                               )
                logger.info('[iteration:{}][map:{:.4f}]'.format(cur_iter, mAP.item()))

            # Save snapshot
            if cur_iter % snapshot_interval == snapshot_interval - 1:
                paddle.save(model.state_dict(), checkpoint)
                logger.info('[iteration:{}][Snapshot]'.format(cur_iter))

    # Evaluate and save snapshot
    mAP = evaluate(logger,
                   model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   topk,
                   multi_labels,
                   test_label,
                   train_label,
                   database_label
                   )
    paddle.save(model.state_dict(), checkpoint)
    logger.info('Training finish, [iteration:{}][map:{:.4f}][Snapshot]'.format(cur_iter, mAP.item()))


def evaluate(logger, model, query_dataloader, retrieval_dataloader, code_length, topk, multi_labels, test_label, train_label, database_label):
    """
    Evaluate.

    Args
        model(paddle.nn.Layer): CNN model.
        query_dataloader(paddle.io.DataLoader): Query data loader.
        retrieval_dataloader(paddle.io.DataLoader): Retrieval data loader.
        code_length(int): Hash code length.
        topk(int): Calculate top k data points map.
        multi_labels(bool): Multi labels.

    Returns
        mAP(float): Mean average precision.
    """
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length)

    # One-hot encode targets
    # onehot_query_targets = test_label
    # onehot_retrieval_targets = database_label

    onehot_query_targets = paddle.to_tensor(np.array(test_label)).cuda()
    onehot_retrieval_targets = paddle.to_tensor(np.array(database_label)).cuda()

    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        topk,
    )

    model.train()

    return mAP


def generate_code(model, dataloader, code_length):
    """
    Generate hash code.

    Args
        model(paddle.nn.Layer): CNN model.
        dataloader(paddle.io.DataLoader): Data loader.
        code_length(int): Hash code length.

    Returns
        code(paddle.Tensor): Hash code.
    """
    with paddle.no_grad():
        N = len(dataloader.dataset)
        code = paddle.zeros([N, code_length])
        for data, _, index in dataloader:
            data = paddle.to_tensor(data).cuda()
            outputs = model(data)
            code[index, :] = paddle.sign(outputs).cpu()

    return code

def generate_similarity_matrix(features, alpha, beta):
    """
    Generate similarity matrix.

    Args
        features(paddle.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(paddle.Tensor): Similarity matrix.
    """
    # Cosine similarity
    cos_dist = squareform(pdist(features.numpy(), 'cosine'))

    # Find maximum count of cosine distance
    max_cnt, max_cos = 0, 0
    interval = 1. / 100
    cur = 0
    for i in range(100):
        cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
        if max_cnt < cur_cnt:
            max_cnt = cur_cnt
            max_cos = cur
        cur += interval

    # Split features into two parts
    flat_cos_dist = cos_dist.reshape((-1, 1))
    left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
    right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

    # Reconstruct gaussian distribution
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([2 * max_cos - right, right])

    # Model data using gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    # Construct similarity matrix
    S = (cos_dist < (left_mean - alpha * left_std)) * 1.0 + (cos_dist > (right_mean + beta * right_std)) * -1.0

    return paddle.to_tensor(S, dtype='float32')


def extract_features(model, dataloader, num_features, verbose):
    """
    Extract features.

    Args
        model(paddle.nn.Layer): CNN model.
        dataloader(paddle.io.DataLoader): Data loader.
        num_features(int): Number of features.
        verbose(bool): Print log.

    Returns
        features(paddle.Tensor): Features.
    """
    model.eval()
    model.set_extract_features(True)
    features = paddle.zeros([dataloader.dataset.imgs.shape[0], num_features])
    with paddle.no_grad():
        for i, (data, _, index) in enumerate(dataloader):
            data = paddle.to_tensor(data, dtype='float32')
            features[index, :] = model(data).cpu()

    model.set_extract_features(False)
    model.train()

    return features


class SSDH_Loss(nn.Layer):
    def __init__(self):
        super(SSDH_Loss, self).__init__()

    def forward(self, H, S):
        loss = (S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)

        return loss