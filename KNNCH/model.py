import sys
sys.path.append('/home1/ljy/paddle/paddle_project/utils')
import paddle
# from layers.GNN import GraphAttention
from utils import *
import paddle.nn.functional as F


class MLP(paddle.nn.Layer):

    def __init__(self, units: list):
        super(MLP, self).__init__()
        self.units = units
        self.hidden_numbers = len(self.units) - 1
        layers = []
        for i in range(self.hidden_numbers):
            layers.extend([paddle.nn.Linear(in_features=self.units[i], out_features=self.units[i + 1]), paddle.nn.BatchNorm1D(num_features=units[i + 1]), paddle.nn.Tanh()])
        self.backbone_net = paddle.nn.LayerList(sublayers=layers)
        self.backbone_net = paddle.nn.Sequential(*self.backbone_net)

    def forward(self, x):
        z = self.backbone_net(x)
        return z

class KNNGenerator(paddle.nn.Layer):

    def __init__(self, K=10, dis_type='L2'):
        super(KNNGenerator, self).__init__()
        self.K = K
        self.dis_type = dis_type

    def distance(self, mat1, mat2, type='cosine'):
        if type == 'cosine':
            mat1_norm = F.normalize(mat1)
            mat2_norm = F.normalize(mat2)
            sim = mat1_norm.mm(mat2=mat2_norm)
            dis = -sim
        elif type == 'L2':
            dis = paddle.cdist(x=mat1, y=mat2, p=2)
        else:
            dis = None
        return dis

    def forward(self, feat, anchor, target_anchor):
        dis = self.distance(feat, anchor, type=self.dis_type)
        index = paddle.argsort(x=dis, axis=1, descending=False)
        index_topk = index[:, :self.K]
        index_topk = index_topk.unsqueeze(-1)
        recons = paddle.mean(paddle.gather_nd(target_anchor, index_topk), axis=1)
        return recons


class FFNGenerator(paddle.nn.Layer):

    def __init__(self, input_dim, output_dim):
        super(FFNGenerator, self).__init__()
        self.mlp = MLP(units=[input_dim, 2048, output_dim])

    def forward(self, feat):
        recons = self.mlp(feat)
        return recons


class Fusion(paddle.nn.Layer):

    def __init__(self, fusion_dim=1024, nbit=64) ->None:
        super(Fusion, self).__init__()
        self.hash = paddle.nn.Sequential(paddle.nn.Linear(in_features=fusion_dim, out_features=nbit), paddle.nn.BatchNorm1D(num_features=nbit), paddle.nn.Tanh())

    def forward(self, x, y):
        hash_code = self.hash(x + y)
        return hash_code
