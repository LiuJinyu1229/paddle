import paddle
from paddle import nn
from paddle.nn import functional as F
from models.basic_module import BasicModule

LAYER1_NODE = 8192

def weights_init(m):
    if isinstance(m, nn.Conv2D):
        m.weight.set_value(paddle.normal(mean=0.0, std=0.01, shape=m.weight.shape))
        m.bias.set_value(paddle.normal(mean=0.0, std=0.01, shape=m.bias.shape))

class TxtModule(BasicModule):
    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        self.conv1 = nn.Conv2D(1, LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2D(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = paddle.squeeze(x)
        return x