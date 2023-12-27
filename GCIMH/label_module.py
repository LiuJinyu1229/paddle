import paddle
import paddle.nn as nn

class LabelModule(nn.Layer):
    def __init__(self, label_dim, n_bits):
        super(LabelModule, self).__init__()

        self.fc1 = nn.Linear(label_dim, 4096)
        self.BN1 = nn.BatchNorm1D(4096)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(4096, 4096)
        self.BN2 = nn.BatchNorm1D(4096)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(4096, n_bits)
        self.BN3 = nn.BatchNorm1D(n_bits)
        self.act3 = nn.Tanh()

    def forward(self, l):
        l = self.fc1(l)
        l = self.BN1(l)
        l = self.act1(l)
        r = l

        l = self.fc2(l)
        l = self.BN2(l)
        l += r
        l = self.act2(l)

        l = self.fc3(l)
        l = self.BN3(l)
        l = self.act3(l)

        return l