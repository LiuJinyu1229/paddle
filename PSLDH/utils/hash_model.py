import paddle
import paddle.nn as nn
import paddle.vision.models as models

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
        if model_name == "alexnet":
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
                cl3,
                nn.Tanh()
            )
            self.model_name = 'alexnet'
        if 'ResNet' in model_name:
            self.model_name = model_name
            resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
                           "ResNet101": models.resnet101, "ResNet152": models.resnet152}
            model_resnet = resnet_dict[model_name](pretrained=True)
            self.conv1 = model_resnet.conv1
            self.bn1 = model_resnet.bn1
            self.relu = model_resnet.relu
            self.maxpool = model_resnet.maxpool
            self.layer1 = model_resnet.layer1
            self.layer2 = model_resnet.layer2
            self.layer3 = model_resnet.layer3
            self.layer4 = model_resnet.layer4
            self.avgpool = model_resnet.avgpool
            self.features = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                                self.layer1, self.layer2, self.layer3, self.layer4,
                                                self.avgpool)
            in_features = model_resnet.fc.weight.shape[1]
            self.classifier = nn.Linear(in_features, bit)
            weight_initializer = nn.initializer.Normal(mean=0, std=0.01)
            weight_initializer(self.classifier.weight)
            self.classifier.bias.set_value(paddle.full(shape=[bit], fill_value=0.0))

            self.activation = nn.Tanh()
            self.mean = paddle.to_tensor([[[0.485]], [[0.456]], [[0.406]]])
            self.std = paddle.to_tensor([[[0.229]], [[0.224]], [[0.225]]])

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            # f = f.view(f.size(0), 256 * 6 * 6)
            f = f.reshape([f.shape[0], 256 * 6 * 6])
            f = self.classifier(f)
        elif 'vgg' in self.model_name:
            f = f.view(f.size(0), -1)
            f = self.classifier(f)
        else:
            # f = f.reshape([f.shape[0], -1])
            f = f.reshape([f.shape[0], -1])
            y = self.classifier(f)
            f = self.activation(y)
        return f