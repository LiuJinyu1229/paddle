import paddle
from paddle import nn
import paddle.nn.initializer as init
import os
from .CNN_F import image_net

class GEN(nn.Layer):
    def __init__(self, dropout, image_dim, text_dim, hidden_dim, output_dim, pretrain_model=None):
        super(GEN, self).__init__()
        self.module_name = 'GEN_module'
        self.output_dim = output_dim
        # self.cnn_f = image_net(pretrain_model)   ## if use 4096-dims feature, pass
        if dropout:
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.BatchNorm1D(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1D(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1D(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.text_module = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.BatchNorm1D(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1D(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1D(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
        else:
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.BatchNorm1D(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1D(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1D(hidden_dim // 4),
                nn.ReLU(),
            )
            self.text_module = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.BatchNorm1D(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1D(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1D(hidden_dim // 4),
                nn.ReLU()
            )

        self.hash_module = nn.LayerDict({
            'image': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Tanh()),
            'text': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Tanh()),
        })


    def weight_init(self):
        initializer = self.kaiming_init
        for block in self._sub_layers:
            if block == 'cnn_f':
                pass
            else:
                for m in self._sub_layers[block]:
                    initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2D)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.set_value(0)
        elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D)):
            m.weight.set_value(1)
            if m.bias is not None:
                m.bias.set_value(0)

    def forward(self, x, y):
        x = x.astype('float32')
        # x = self.cnn_f(x).squeeze()   ## if use 4096-dims feature, pass
        f_x = self.image_module(x)
        f_y = self.text_module(y)

        x_code = self.hash_module['image'](f_x).reshape([-1, self.output_dim])
        y_code = self.hash_module['text'](f_y).reshape([-1, self.output_dim])
        return x_code, y_code, paddle.squeeze(f_x), paddle.squeeze(f_y)

    def generate_img_code(self, i):
        # i = self.cnn_f(i).squeeze()   ## if use 4096-dims feature, pass
        f_i = self.image_module(i)

        code = self.hash_module['image'](f_i.detach()).reshape([-1, self.output_dim])
        return code

    def generate_txt_code(self, t):
        f_t = self.text_module(t)

        code = self.hash_module['text'](f_t.detach()).reshape([-1, self.output_dim])
        return code

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.set_state_dict(paddle.load(path, map_location='cpu'))
        else:
            self.set_state_dict(paddle.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        paddle.save(self.state_dict(), os.path.join(path, name))
        return name
