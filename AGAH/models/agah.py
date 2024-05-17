import paddle
from paddle import nn

class AGAH(nn.Layer):
    def __init__(self, bit, y_dim, num_label, emb_dim, lambd=0.8, pretrain_model=None):
        super(AGAH, self).__init__()
        self.module_name = 'AGAH'
        self.bit = bit
        self.lambd = lambd

        self.img_module = nn.Sequential(
            nn.Conv2D(3, 64, 11, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm2D(64),
            nn.Pad2D(padding=[0, 1, 0, 1]),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),

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

            nn.Conv2D(256, 4096, 6, stride=1),
            nn.ReLU(),

            nn.Conv2D(4096, 4096, 1, stride=1),
            nn.ReLU(),

            nn.Conv2D(4096, emb_dim, 1),
            nn.ReLU()
        )

        self.txt_module = nn.Sequential(
            nn.Conv2D(1, 8192, kernel_size=(y_dim, 1), stride=1),
            nn.ReLU(),
            nn.Conv2D(8192, 4096, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2D(4096, emb_dim, 1),
            nn.ReLU()
        )

        self.hash_module = nn.LayerDict({
            'img': nn.Sequential(
                nn.Conv2D(emb_dim, bit, 1),
                nn.Tanh()
            ),
            'txt': nn.Sequential(
                nn.Conv2D(emb_dim, bit, 1),
                nn.Tanh()
            )
        })

        self.classifier = nn.LayerDict({
            'img': nn.Sequential(
                nn.Conv2D(emb_dim, num_label, 1),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                nn.Conv2D(emb_dim, num_label, 1),
                nn.Sigmoid()
            ),
        })

        self.img_discriminator = nn.Sequential(
            nn.Conv2D(1, emb_dim, kernel_size=(emb_dim, 1)),
            nn.ReLU(),

            nn.Conv2D(emb_dim, 256, 1),
            nn.ReLU(),

            nn.Conv2D(256, 1, 1)
        )

        self.txt_discriminator = nn.Sequential(
            nn.Conv2D(1, emb_dim, kernel_size=(emb_dim, 1)),
            nn.ReLU(),

            nn.Conv2D(emb_dim, 256, 1),
            nn.ReLU(),

            nn.Conv2D(256, 1, 1)
        )

        if pretrain_model is not None:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        for i, v in self.img_module.named_children():
            k = int(i)
            if k >= 20:
                break
            if isinstance(v, nn.Conv2D):
                if k > 1:
                    k -= 1
                v.weight.set_value(paddle.to_tensor(weights[k][0][0][0][0][0].transpose()))
                v.bias.set_value(paddle.to_tensor(weights[k][0][0][0][0][1].reshape(-1)))

    def forward(self, x, y, feature_map=None):
        f_x = self.img_module(x)
        f_y = self.txt_module(y.unsqueeze(1).unsqueeze(-1))

        # normalization
        f_x = f_x / paddle.sqrt(paddle.sum(f_x.detach() ** 2))
        f_y = f_y / paddle.sqrt(paddle.sum(f_y.detach() ** 2))

        # attention
        if feature_map is not None:
            # img attention
            mask_img = paddle.nn.functional.sigmoid(5 * f_x.squeeze().matmul(feature_map.t()))  # size: (batch, num_label)
            mask_f_x = mask_img.matmul(feature_map) / paddle.sum(mask_img, axis=1).unsqueeze(-1)  # size: (batch, emb_dim)
            mask_f_x = self.lambd * f_x + (1 - self.lambd) * mask_f_x.unsqueeze(-1).unsqueeze(-1)

            # txt attention
            mask_txt = paddle.nn.functional.sigmoid(5 * f_y.squeeze().matmul(feature_map.t()))
            mask_f_y = mask_txt.matmul(feature_map) / paddle.sum(mask_txt, axis=1).unsqueeze(-1)
            mask_f_y = self.lambd * f_y + (1 - self.lambd) * mask_f_y.unsqueeze(-1).unsqueeze(-1)
        else:
            mask_f_x, mask_f_y = f_x, f_y

        x_class = self.classifier['img'](mask_f_x).squeeze()
        y_class = self.classifier['txt'](mask_f_y).squeeze()
        x_code = self.hash_module['img'](mask_f_x).reshape([-1, self.bit])
        y_code = self.hash_module['txt'](mask_f_y).reshape([-1, self.bit])
        return x_code, y_code, f_x.squeeze(), f_y.squeeze(), x_class, y_class

    def dis_img(self, f_x):
        is_img = self.img_discriminator(f_x.unsqueeze(1).unsqueeze(-1))
        return is_img.squeeze()

    def dis_txt(self, f_y):
        is_txt = self.txt_discriminator(f_y.unsqueeze(1).unsqueeze(-1))
        return is_txt.squeeze()

    def generate_img_code(self, x, feature_map=None):
        f_x = self.img_module(x)
        f_x = f_x / paddle.sqrt(paddle.sum(f_x.detach() ** 2))

        # attention
        if feature_map is not None:
            mask_img = paddle.nn.functional.sigmoid(5 * f_x.squeeze().matmul(feature_map.t()))  # size: (batch, num_label)
            mask_f_x = mask_img.matmul(feature_map) / paddle.sum(mask_img, axis=1).unsqueeze(-1)
            f_x = self.lambd * f_x + (1 - self.lambd) * mask_f_x.unsqueeze(-1).unsqueeze(-1)

        code = self.hash_module['img'](f_x).reshape([-1, self.bit])
        return code

    def generate_txt_code(self, y, feature_map=None):
        f_y = self.txt_module(y.unsqueeze(1).unsqueeze(-1))
        f_y = f_y / paddle.sqrt(paddle.sum(f_y.detach() ** 2))

        # attention
        if feature_map is not None:
            mask_txt = paddle.nn.functional.sigmoid(5 * f_y.squeeze().matmul(feature_map.t()))
            mask_f_y = mask_txt.matmul(feature_map) / paddle.sum(mask_txt, axis=1).unsqueeze(-1)
            f_y = self.lambd * f_y + (1 - self.lambd) * mask_f_y.unsqueeze(-1).unsqueeze(-1)

        code = self.hash_module['txt'](f_y).reshape([-1, self.bit])
        return code


