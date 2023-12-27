import paddle
from paddle import nn
import paddle.nn.initializer as init
from paddle.nn import LayerDict
from paddle.nn.functional import interpolate
import paddle.vision as vision
import paddle.nn.functional as function
from config import opt


class image_net(nn.Layer):
    def __init__(self, pretrain_model):
        super(image_net, self).__init__()
        self.img_module = nn.Sequential(
            # 0 conv1
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=11, stride=(4, 4), padding=(0, 0)),
            # 1 relu1
            nn.ReLU(),
            # 2 norm1
            nn.BatchNorm2D(64),
            # 3 pool1
            nn.Pad2D(padding=[0, 1, 0, 1]),
            nn.MaxPool2D(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),

            # 4 conv2
            nn.Conv2D(in_channels=64, out_channels=256, kernel_size=5, stride=(1, 1), padding=(2, 2)),
            # 5 relu2
            nn.ReLU(),
            # 6 norm2
            nn.BatchNorm2D(256),
            # 7 pool2
            nn.MaxPool2D(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),

            # 8 conv3
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 9 relu3
            nn.ReLU(),

            # 10 conv4
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 11 relu4
            nn.ReLU(),
            # 12 conv5
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 13 relu5
            nn.ReLU(),
            # 14 pool5
            nn.MaxPool2D(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2D(in_channels=256, out_channels=4096, kernel_size=6, stride=(1, 1)),
            # 16 relu6
            nn.ReLU(),
            # 17 full_conv7
            nn.Conv2D(in_channels=4096, out_channels=4096, kernel_size=1, stride=(1, 1)),
            # 18 relu7
            nn.ReLU()
            # 19 full_conv8
        )
        self.mean = paddle.zeros([3, 224, 224])
        self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = paddle.to_tensor(data['normalization'][0][0][0].transpose()).astype('float32')
        for i, v in enumerate(self.img_module.sublayers()):
            k = int(i)
            if k >= 20:
                break
            if isinstance(v, nn.Conv2D):
                if k > 1:
                    k -= 1
                v.weight.set_value(paddle.to_tensor(weights[k][0][0][0][0][0].transpose()))
                v.bias.set_value(paddle.to_tensor(weights[k][0][0][0][0][1].reshape(-1)))
        print('sucusses init!')

    def forward(self, x):
        x = x - self.mean
        f_x = self.img_module(x)
        return f_x