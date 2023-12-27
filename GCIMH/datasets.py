import paddle.io

class ImgFile(paddle.io.Dataset):
    def __init__(self, I, T, L, M1, M2):
        self.images = paddle.to_tensor(I)
        self.tags = paddle.to_tensor(T)
        self.labels = paddle.to_tensor(L)
        self.m1s = paddle.to_tensor(M1)
        self.m2s = paddle.to_tensor(M2)
        self.length = L.shape[0]

    def __getitem__(self, index):
        img = self.images[index]
        tag = self.tags[index]
        label = self.labels[index]
        m1 = self.m1s[index]
        m2 = self.m2s[index]

        return img, tag, label, m1, m2

    def __len__(self):
        return self.length