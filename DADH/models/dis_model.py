import paddle
from paddle import nn
import paddle.nn.initializer as init
import numpy as np

class DIS(nn.Layer):
    def __init__(self, input_dim, hidden_dim, hash_dim):
        super(DIS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        self.feature_dis = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1)
        )

        self.hash_dis = nn.Sequential(
            nn.Linear(self.hash_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.weight_init()


    def weight_init(self):
        for block in self._sub_layers:
            for name, m in self._sub_layers[block].named_sublayers():
                self.kaiming_init(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2D)):
            init.KaimingNormal()(m.weight)
            if m.bias is not None:
                m.bias.set_value(np.zeros_like(m.bias.numpy()))
        elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D)):
            m.weight.set_value(np.ones_like(m.weight.numpy()))
            if m.bias is not None:
                m.bias.set_value(np.zeros_like(m.bias.numpy()))

    def dis_feature(self, f):
        feature_score = self.feature_dis(f)
        return paddle.squeeze(feature_score)

    def dis_hash(self, h):
        hash_score = self.hash_dis(h)
        return paddle.squeeze(hash_score)