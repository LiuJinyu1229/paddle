import paddle
import math
from collections import OrderedDict


class PositionalEncoding(paddle.nn.Layer):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    Refer:
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = paddle.nn.Dropout(p=dropout)
        pe = paddle.zeros(shape=[max_len, d_model])
        position = paddle.arange(start=0, end=max_len, dtype='float32'
            ).unsqueeze(axis=1)
        div_term = paddle.exp(x=paddle.arange(start=0, end=d_model, step=2)
            .astype(dtype='float32') * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(x=position * div_term)
        pe[:, 1::2] = paddle.cos(x=position * div_term)
        x = pe.unsqueeze(axis=0)
        perm_0 = list(range(x.ndim))
        perm_0[0] = 1
        perm_0[1] = 0
        pe = x.transpose(perm=perm_0)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class MLP(paddle.nn.Layer):

    def __init__(self, act, hidden_dim=[1000, 2048, 512]):
        super(MLP, self).__init__()
        self.input_dim = hidden_dim[0]
        self.hidden_dim = hidden_dim
        self.mlp = paddle.nn.Sequential()
        # orderedDict = OrderedDict()
        for i in range(len(hidden_dim) - 1):
            index = i + 1
            self.mlp.add_sublayer('linear' + str(index), paddle.nn.Linear(in_features=self.hidden_dim[i], out_features=self.hidden_dim[i + 1]))
            self.mlp.add_sublayer('bn' + str(index), paddle.nn.BatchNorm1D(num_features=self.hidden_dim[i + 1]))
            if act == 'gelu':
                self.mlp.add_sublayer('act' + str(index), paddle.nn.GELU())
            elif act == 'tanh':
                self.mlp.add_sublayer('act' + str(index), paddle.nn.Tanh())

    def _initialize(self):
        init_XavierNormal = paddle.nn.initializer.XavierNormal()
        init_XavierNormal(self.mlp.linear1.weight.data)
        init_XavierNormal = paddle.nn.initializer.XavierNormal()
        init_XavierNormal(self.mlp.linear2.weight.data)

    def forward(self, x):
        return self.mlp(x)
