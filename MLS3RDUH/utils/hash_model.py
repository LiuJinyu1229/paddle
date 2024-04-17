import paddle.nn as nn
from paddle.vision import models
import paddle

LAYER1_NODE = 40960

def weights_init(m):
    if isinstance(m, nn.Conv2D):
        nn.initializer.XavierUniform(m.weight)
        nn.initializer.Constant(m.bias, 0.)

class AlexNet(nn.Layer):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x

class HASH_Net(nn.Layer):
    def __init__(self, model_name, bit, pretrained=True):
        super(HASH_Net, self).__init__()
        original_model = AlexNet()
        self.features = original_model.features
        # self.features_i2t = original_model.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl2 = nn.Linear(4096, 4096)
        cl3 = nn.Linear(4096, bit)
        if pretrained:
            cl1.weight.set_value(original_model.classifier[1].weight.numpy())
            cl1.bias.set_value(original_model.classifier[1].bias.numpy())
            cl2.weight.set_value(original_model.classifier[4].weight.numpy())
            cl2.bias.set_value(original_model.classifier[4].bias.numpy())
        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(),
            nn.Dropout(),
            cl2,
            nn.ReLU(),
        )
        self.hash = nn.Sequential(
            cl3,
            nn.Tanh()
        )
        self.model_name = 'alexnet'

    def forward(self, x):
        f = self.features(x)
        f = f.reshape([f.shape[0], 256 * 6 * 6])
        f = self.classifier(f)
        y = self.hash(f)
        return f, y

