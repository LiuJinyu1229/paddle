import paddle
from paddle import nn
from paddle.nn import functional as F

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.initializer.XavierUniform(m.weight)
        nn.initializer.Constant(m.bias, 0.00)

class Label_net(nn.Layer):
    def __init__(self, label_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(Label_net, self).__init__()
        self.module_name = "text_model"
        # 400
        cl1 = nn.Linear(label_dim, 512)
        cl2 = nn.Linear(512, bit)
        self.cl_text = nn.Sequential(
            cl1,
            nn.ReLU(),
            cl2,
            nn.Tanh()
        )

    def forward(self, x):
        y = self.cl_text(x)
        return y