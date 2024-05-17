import paddle
import paddle.nn as nn
import paddle.vision.models as models
import os

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

def load_model(arch, code_length):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(paddle.nn.Layer): CNN model.
    """
    if arch == 'alexnet':
        model = AlexNet()
        model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, 4096, code_length)
    else:
        raise ValueError("Invalid model name!")

    return model


class ModelWrapper(nn.Layer):
    """
    Add tanh activate function into model.

    Args
        model(paddle.nn.Layer): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )

        # Extract features
        self.extract_features = False

    def forward(self, x):
        if self.extract_features:
            return self.model(x)
        else:
            return self.hash_layer(self.model(x))

    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag

    def snapshot(self, optimizer, checkpoint):
        """
        Save model snapshot.

        Args
            it(int): Iteration.
            optimizer(paddle.optimizer.Optimizer): Optimizer.

        Returns
            None
        """
        paddle.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint)

    def load_snapshot(self, root, optimizer=None):
        """
        Load model snapshot.

        Args
            root(str): Path of model snapshot.
            optimizer(paddle.optimizer.Optimizer): Optimizer.

        Returns
            optimizer(paddle.optimizer.Optimizer, optional): Optimizer, if parameter 'optimizer' given.
            it(int): Iteration, if parameter 'optimizer' given.
        """
        checkpoint = paddle.load(root)
        self.set_state_dict(checkpoint)
