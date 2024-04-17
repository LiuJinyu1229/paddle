import paddle
import time

class BasicModule(paddle.nn.Layer):
    """
    封装nn.Layer，主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            model_state_dict = paddle.load(path)
            self.set_state_dict(model_state_dict)
        else:
            model_state_dict = paddle.load(path)
            self.set_state_dict(model_state_dict)

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pdparams')
        paddle.save(self.state_dict(), 'model/' + name)
        return name

    def forward(self, *input):
        pass