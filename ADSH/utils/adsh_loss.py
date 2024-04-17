import paddle
import paddle.nn as nn

class ADSHLoss(nn.Layer):
    def __init__(self, gamma, code_length, num_train):
        super(ADSHLoss, self).__init__()
        self.gamma = gamma
        self.code_length = code_length
        self.num_train = num_train

    def forward(self, u, V, S, V_omega):
        batch_size = u.shape[0]
        V = paddle.to_tensor(V).astype('float32')
        V_omega = paddle.to_tensor(V_omega).astype('float32')
        S = S.astype('float32')
        square_loss = (paddle.matmul(u, paddle.transpose(V, [1, 0]))-self.code_length * S) ** 2
        quantization_loss = self.gamma * (V_omega - u) ** 2
        loss = (square_loss.sum() + quantization_loss.sum()) / (self.num_train * batch_size)
        return loss