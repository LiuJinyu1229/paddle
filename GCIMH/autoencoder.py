import paddle.nn as nn
import paddle

class AutoEncoderGcnModule(nn.Layer):
    def __init__(self, image_dim, text_dim):
        super(AutoEncoderGcnModule, self).__init__()
        fusion_dim = image_dim + text_dim
        self.encoder1_gconv = nn.Linear(fusion_dim, 6144)
        self.encoder1_bn = nn.BatchNorm1D(6144)
        self.encoder1_act = nn.ReLU()

        self.encoder2_gconv = nn.Linear(6144, 6144)
        self.encoder2_bn = nn.BatchNorm1D(6144)
        self.encoder2_act = nn.ReLU()

        self.encoder3_gconv = nn.Linear(6144, 8192)
        self.encoder3_bn = nn.BatchNorm1D(8192)
        self.encoder3_act = nn.ReLU()

        self.decoder1_gconv = nn.Linear(8192, 6144)
        self.decoder1_bn = nn.BatchNorm1D(6144)
        self.decoder1_act = nn.ReLU()

        self.decoder2_gconv = nn.Linear(6144, 6144)
        self.decoder2_bn = nn.BatchNorm1D(6144)
        self.decoder2_act = nn.ReLU()

        self.image_gconv = nn.Linear(6144, image_dim)
        self.image_bn = nn.BatchNorm1D(image_dim)

        self.text_gconv = nn.Linear(6144, text_dim)
        self.text_bn = nn.BatchNorm1D(text_dim)
        self.text_act = nn.Sigmoid()

    def forward(self, graph, e):
        e = self.encoder1_gconv(e)
        e = paddle.mm(graph, e)
        e = self.encoder1_bn(e)
        e = self.encoder1_act(e)
        r = e

        e = self.encoder2_gconv(e)
        e = paddle.mm(graph, e)
        e = self.encoder2_bn(e)
        e += r
        e = self.encoder2_act(e)

        e = self.encoder3_gconv(e)
        e = paddle.mm(graph, e)
        e = self.encoder3_bn(e)
        e = self.encoder3_act(e)

        d = self.decoder1_gconv(e)
        d = paddle.mm(graph, d)
        d = self.decoder1_bn(d)
        d = self.decoder1_act(d)
        r = d

        d = self.decoder2_gconv(d)
        d = paddle.mm(graph, d)
        d = self.decoder2_bn(d)
        d += r
        d = self.decoder2_act(d)

        i = self.image_gconv(d)
        i = paddle.mm(graph, i)
        i = self.image_bn(i)

        t = self.text_gconv(d)
        t = paddle.mm(graph, t)
        t = self.text_bn(t)
        t = self.text_act(t)

        return i, t, e
