import paddle.nn as nn
from paddle.vision import models
import paddle

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

class cnn_model(nn.Layer):
    def __init__(self, model_name, bit):
        super(cnn_model, self).__init__()
        original_model = AlexNet()
        self.features = original_model.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight.set_value(original_model.classifier[1].weight.numpy())
        cl1.bias.set_value(original_model.classifier[1].bias.numpy())

        cl2 = nn.Linear(4096, 4096)
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
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, bit),
            nn.Tanh()
        )
        self.model_name = model_name


    def forward(self, x):
        f = self.features(x)
        f = f.reshape([f.shape[0], 256 * 6 * 6])
        y = self.classifier(f)
        code = self.hash_layer(y)
        return y, code
