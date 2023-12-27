import paddle
import paddle.nn as nn

class FusionModule(nn.Layer):
    def __init__(self, fusion_dim, n_bits):
        super(FusionModule, self).__init__()

        self.fc1 = nn.Linear(fusion_dim, 8192)
        self.BN1 = nn.BatchNorm1D(8192)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(8192, 4096)
        self.BN2 = nn.BatchNorm1D(4096)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(4096, n_bits)
        self.BN3 = nn.BatchNorm1D(n_bits)
        self.act3 = nn.Tanh()

    def forward(self, f1):
        f1 = self.fc1(f1)
        f1 = self.BN1(f1)
        f1 = self.act1(f1)

        f2 = self.fc2(f1)
        f2 = self.BN2(f2)
        f2 = self.act2(f2)

        f2 = self.fc3(f2)
        f2 = self.BN3(f2)
        f2 = self.act3(f2)

        return f1, f2