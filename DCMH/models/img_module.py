import paddle
from paddle import nn
from models.basic_module import BasicModule
import paddle.nn.initializer as init

class ImgModule(BasicModule):
    def __init__(self, bit, pretrain_model=None):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.features = nn.Sequential(
            nn.Conv2D(3, 64, 11, stride=4),
            nn.ReLU(),
            nn.BatchNorm2D(64),
            nn.Pad2D(padding=[0, 1, 0, 1]),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(64, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2D(256),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(256, 4096, 6),
            nn.ReLU(),
            nn.Conv2D(4096, 4096, 1),
            nn.ReLU(),
        )
        self.classifier = paddle.nn.Linear(in_features=4096, out_features=bit)
        self.classifier.weight.set_value(paddle.randn([4096, bit]) * 0.01)
        self.classifier.bias.set_value(paddle.randn([bit]) * 0.01)
        self.mean = paddle.zeros([3, 224, 224])
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = paddle.to_tensor(data['normalization'][0][0][0].transpose()).astype('float32')
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2D):
                if k > 1:
                    k -= 1
                v.weight.set_value(paddle.to_tensor(weights[k][0][0][0][0][0].transpose()))
                v.bias.set_value(paddle.to_tensor(weights[k][0][0][0][0][1].reshape(-1)))

    def forward(self, x):
        x = x - self.mean
        x = self.features(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.classifier(x)
        return x