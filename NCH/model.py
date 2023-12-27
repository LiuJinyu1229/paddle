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


# class GCNEncoder(paddle.nn.Layer):

#     def __init__(self, input_dim, generate_dim, edge_dim, dropout=0.5, beta=1
#         ) ->None:
#         super(GCNEncoder, self).__init__()
#         self.input_dim = input_dim
#         self.generate_dim = generate_dim
#         self.edge_dim = edge_dim
#         self.beta = beta
#         self.gcn = GraphAttention(self.input_dim, self.generate_dim, self.
#             edge_dim, dropout)

#     def forward(self, param_dict: dict):
#         recons = self.gcn(**param_dict)
#         return recons * self.beta


# class KNNGenerator(paddle.nn.Layer):

#     def __init__(self, K=10, dis_type='L2'):
#         super(KNNGenerator, self).__init__()
#         self.K = K
#         self.dis_type = dis_type

#     def distance(self, mat1, mat2, type='cosine'):
#         if type == 'cosine':
#             mat1_norm = F.normalize(mat1)
#             mat2_norm = F.normalize(mat2)
#             sim = mat1_norm.mm(mat2=mat2_norm)
#             dis = -sim
#         elif type == 'L2':
#             dis = paddle.cdist(x=mat1, y=mat2, p=2)
#         else:
#             dis = None
#         return dis

#     def forward(self, feat, anchor, target_anchor):
#         dis = self.distance(feat, anchor, type=self.dis_type)
#         index = paddle.argsort(x=dis, axis=1, descending=False)
#         index_topk = index[:, :self.K]
#         recons = paddle.mean(x=target_anchor[index_topk], axis=1)
#         return recons


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


class SelfAttention(paddle.nn.Layer):

    def __init__(self, Q_dim, K_dim, V_dim, d_model) ->None:
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.Q_layer = paddle.nn.Linear(in_features=Q_dim, out_features=d_model)
        self.K_layer = paddle.nn.Linear(in_features=K_dim, out_features=d_model)
        self.V_layer = paddle.nn.Linear(in_features=V_dim, out_features=d_model)

    def forward(self, Q, K, V, label_graph):
        Q = self.Q_layer(Q)
        K = self.Q_layer(K)
        V = self.V_layer(V)
        attention_score = Q @ K.t() / math.sqrt(self.d_model)
        # import ipdb
        # ipdb.set_trace()
        if label_graph.shape[0] != 0:
            attention_score = paddle.multiply(label_graph, attention_score)
            # attention_score = label_graph.mul(attention_score)
        attention_prob = F.softmax(attention_score, axis=-1)
        context = attention_prob @ V
        return context, attention_prob


class TransformerEncoder(paddle.nn.Layer):

    def __init__(self, Q_dim, K_dim, V_dim, d_model=1024, dim_feedforward=
        2048, dropout=0.5) ->None:
        super(TransformerEncoder, self).__init__()
        self.self_attn = SelfAttention(Q_dim, K_dim, V_dim, d_model)
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=dim_feedforward)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(in_features=dim_feedforward,out_features=d_model)
        self.norm1 = paddle.nn.BatchNorm1D(num_features=d_model)
        self.norm2 = paddle.nn.BatchNorm1D(num_features=d_model)
        self.dropout1 = paddle.nn.Dropout(p=dropout)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.decoder = paddle.nn.Sequential(paddle.nn.Linear(in_features=d_model, out_features=V_dim), paddle.nn.BatchNorm1D(num_features=V_dim), paddle.nn.Tanh())
        self.activation = paddle.nn.Tanh()

    def forward(self, src, anchor_1, anchor_2, label_graph):
        src2, _ = self.self_attn(src, anchor_1, anchor_2, label_graph)
        src = src2
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = self.decoder(src)
        return src
