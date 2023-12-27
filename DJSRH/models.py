import paddle
import math
import paddle.vision.models as models
import paddle.nn as nn
import paddle.nn.functional as F

class AlexNet(paddle.nn.Layer):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = paddle.nn.Sequential(
            paddle.nn.Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(64, 192, kernel_size=5, padding=2),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(192, 384, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(384, 256, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(256, 256, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.avgpool = paddle.nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = paddle.nn.Sequential(
            paddle.nn.Dropout(),
            paddle.nn.Linear(256 * 6 * 6, 4096),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(),
            paddle.nn.Linear(4096, 4096),
            paddle.nn.ReLU(),
            paddle.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x

class ImgNet(nn.Layer):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        # self.alexnet = models.alexnet(pretrained=True)
        self.alexnet = AlexNet()
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.alexnet.features(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        code = paddle.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Layer):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = x.astype('float32')
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = paddle.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)