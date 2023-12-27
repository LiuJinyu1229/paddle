import paddle

def negative_log_likelihood_similarity_loss0(u, v, s):
    u = u.astype('float64')
    v = v.astype('float64')
    omega = paddle.matmul(u, v.T) / 2
    loss = -((s > 0).astype('float32') * omega - paddle.log(1 + paddle.exp(omega)))
    loss = paddle.mean(loss)
    return loss


def negative_log_likelihood_similarity_loss1(u, v, s, bit):
    u = u.astype('float64')
    v = v.astype('float64')
    omega = paddle.matmul(u, v.t()) / (bit / 18)
    loss = -((s > 0).astype('float32') * omega - paddle.log(1 + paddle.exp(omega)))
    loss = paddle.mean(loss)
    return loss


def similarity_loss(outputs1, outputs2, similarity):
    loss = (2 * similarity - 1) - paddle.matmul(outputs1, outputs2.t()) / outputs1.shape[1]
    loss = paddle.mean(loss ** 2)
    return loss


def quantization_loss(outputs):
    loss = outputs - paddle.sign(outputs)
    loss = paddle.mean(loss ** 2)
    return loss


def quantization_loss1(outputs):
    BCELoss = paddle.nn.BCELoss()
    loss = BCELoss((outputs + 1) / 2, (paddle.sign(outputs) + 1) / 2)
    return loss


def correspondence_loss(outputs_x, outputs_y):
    loss = outputs_x - outputs_y
    loss = paddle.mean(loss ** 2)
    return loss
