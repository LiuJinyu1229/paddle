import paddle.nn as nn
from paddle.vision import models
import paddle

import paddle.nn.functional as F

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
        self.model_name = model_name

    def forward(self, x):
        f = self.features(x)
        f = f.reshape([f.shape[0], 256 * 6 * 6])
        f = self.classifier(f)
        return f

class TxtNet(nn.Layer):
    def __init__(self, txt_feat_len, code_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc_encode.weight.set_value(paddle.normal(mean=0.0, std=0.3, shape=self.fc_encode.weight.shape))

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        code = self.fc_encode(self.dropout(feat))
        code = paddle.tanh(code)

        return code


class LabelNet(nn.Layer):
    def __init__(self, txt_feat_len, code_len):
        super(LabelNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc1.weight.set_value(paddle.normal(mean=0.0, std=0.2, shape=self.fc1.weight.shape))
        self.fc_encode.weight.set_value(paddle.normal(mean=0.0, std=0.2, shape=self.fc_encode.weight.shape))

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = F.normalize(hid)

        return code


class Label_text_Net(nn.Layer):
    def __init__(self, text_len, nclass, bit, pretrained=True):
        super(Label_text_Net, self).__init__()
        self.fc1 = nn.Linear(text_len, 4096)
        self.fc_encode = nn.Linear(4096, 512)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc_encode.weight.set_value(paddle.normal(mean=0.0, std=0.3, shape=self.fc_encode.weight.shape))

        self.fcl1 = nn.Linear(nclass, 512)
        self.fcl_encode = nn.Linear(512, 512)

        self.dropout1 = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU()

        self.code_layer = nn.Linear(512, bit)
        self.model_name = 'alexnet'

    def forward(self, x, l):
        f = self.relu(self.fc1(x))
        code_t = self.fc_encode(self.dropout(f))
        code_t = F.normalize(code_t)
        #
        y = self.relu1(self.fcl1(l))
        cond_l = self.fcl_encode(self.dropout1(y))
        cond_l = F.normalize(cond_l)
        code = self.code_layer(code_t + cond_l)
        code = F.normalize(code)
        return code


class Label_Net(nn.Layer):
    def __init__(self, model_name, nclass, bit, pretrained=True):
        super(Label_Net, self).__init__()
        original_model = AlexNet()
        self.features = original_model.features
        # self.features_i2t = original_model.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl2 = nn.Linear(4096, 4096)
        self.cl3 = nn.Linear(4096, 512)
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
            # cl3,
            # nn.Tanh()
        )
        self.fcl1 = nn.Linear(nclass, 512)
        self.fcl_encode = nn.Linear(512, 512)

        self.dropout1 = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU()

        self.code_layer = nn.Linear(512, bit)
        self.model_name = model_name

    def forward(self, x, l):
        f = self.features(x)
        f = f.reshape([f.shape[0], 256 * 6 * 6])

        f = self.cl3(self.classifier(f))
        code_v = F.normalize(f)

        y = self.relu1(self.fcl1(l))
        cond_l = self.fcl_encode(self.dropout1(y))
        cond_l = F.normalize(cond_l)
        code = self.code_layer(code_v + cond_l)
        code = F.normalize(code)
        return code
