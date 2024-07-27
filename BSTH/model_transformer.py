import sys
import paddle
import layers

from paddle.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

class ImageTransformerEncoder(paddle.nn.Layer):
    def __init__(self, common_dim, nhead, num_layer, dropout, act):
        super(ImageTransformerEncoder, self).__init__()
        self.imageEncoderLayer = TransformerEncoderLayer(d_model=common_dim,
                                                         nhead=nhead,
                                                         dim_feedforward=common_dim,
                                                         activation=act,
                                                         dropout=dropout)
        self.imageEncoderNorm = LayerNorm(normalized_shape=common_dim)
        self.imageTransformerEncoder = TransformerEncoder(encoder_layer=self.imageEncoderLayer, num_layers=num_layer, norm=self.imageEncoderNorm)

    def forward(self, src):
        output = self.imageTransformerEncoder(src)
        return output

class GMMH(paddle.nn.Layer):

    def __init__(self, args):
        super(GMMH, self).__init__()
        self.image_dim = args.image_dim
        self.text_dim = args.text_dim
        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        self.nbit = int(args.nbit)
        self.classes = args.classes
        self.batch_size = 0
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]
        self.nhead = args.nhead
        # if args.trans_act == 'gelu':
        #     self.act = paddle.nn.GELU()
        # elif args.trans_act == 'tanh':
        #     self.act = paddle.nn.Tanh()
        self.act = args.trans_act
        self.dropout = args.dropout
        self.num_layer = args.num_layer
        self.imageMLP = layers.MLP(hidden_dim=self.img_hidden_dim, act=self.act)
        self.textMLP = layers.MLP(hidden_dim=self.txt_hidden_dim, act=self.act)
        self.imageConcept = paddle.nn.Linear(in_features=self.common_dim,out_features=self.common_dim * self.nbit)
        self.textConcept = paddle.nn.Linear(in_features=self.common_dim,out_features=self.common_dim * self.nbit)
        self.imagePosEncoder = layers.PositionalEncoding(d_model=self.common_dim,dropout=self.dropout)
        self.textPosEncoder = layers.PositionalEncoding(d_model=self.common_dim,dropout=self.dropout)
        imageEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim, 
                                                    nhead=self.nhead, 
                                                    dim_feedforward=self.common_dim,
                                                    activation=self.act, 
                                                    dropout=self.dropout)
        imageEncoderNorm = paddle.nn.LayerNorm(normalized_shape=self.common_dim)
        self.imageTransformerEncoder = TransformerEncoder(encoder_layer=imageEncoderLayer, 
                                                          num_layers=self.num_layer,
                                                          norm=imageEncoderNorm)
        textEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim, 
                                                            nhead=self.nhead, 
                                                            dim_feedforward=self.common_dim,
                                                            activation=self.act, 
                                                            dropout=self.dropout)
        textEncoderNorm = paddle.nn.LayerNorm(normalized_shape=self.common_dim)
        self.textTransformerEncoder = TransformerEncoder(encoder_layer=textEncoderLayer, 
                                                         num_layers=self.num_layer, 
                                                         norm=textEncoderNorm)
        self.hash = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=self.nbit * self.common_dim, out_channels=self.nbit * self.common_dim // 2, kernel_size=1, groups=self.nbit), 
            paddle.nn.BatchNorm2D(num_features=self.nbit * self.common_dim // 2),
            paddle.nn.Tanh(), 
            paddle.nn.Conv2D(in_channels=self.nbit * self.common_dim // 2, out_channels=self.nbit, kernel_size=1, groups=self.nbit), 
            paddle.nn.Tanh())
        self.classify = paddle.nn.Linear(in_features=self.nbit,
            out_features=self.classes)

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                init_XavierNormal = paddle.nn.initializer.XavierNormal()
                init_XavierNormal(p)

    def forward(self, image, text, tgt=None):
        self.batch_size = len(image)
        image = image.astype('float32')
        text = text.astype('float32')
        imageH = self.imageMLP(image)
        textH = self.textMLP(text)

        imageC = self.imageConcept(imageH).reshape([imageH.shape[0], self.nbit, self.common_dim]).transpose(perm=[1, 0, 2])
        textC = self.textConcept(textH).reshape([textH.shape[0], self.nbit,self.common_dim]).transpose(perm=[1, 0, 2])
        
        imageSrc = self.imagePosEncoder(imageC)
        textSrc = self.textPosEncoder(textC)
        
        imageMemory = self.imageTransformerEncoder(imageSrc)
        textMemory = self.textTransformerEncoder(textSrc)
        memory = imageMemory + textMemory
        
        code = self.hash(memory.transpose(perm=[1, 0, 2]).reshape([self.batch_size, self.nbit * self.common_dim, 1, 1])).squeeze()
        return code, self.classify(code)


class L2H_Prototype(paddle.nn.Layer):

    def __init__(self, args):
        super(L2H_Prototype, self).__init__()
        self.classes = args.classes
        self.nbit = args.nbit
        self.d_model = args.nbit
        self.num_layer = 1
        self.nhead = 1
        self.batch_size = 0
        self.labelEmbedding = paddle.nn.Embedding(num_embeddings=self.classes + 1, embedding_dim=self.d_model, padding_idx=0)
        labelEncoderLayer = TransformerEncoderLayer(d_model=self.d_model, 
                                                    nhead=self.nhead, 
                                                    dim_feedforward=self.d_model,
                                                    activation='gelu', 
                                                    dropout=0.5)
        labelEncoderNorm = paddle.nn.LayerNorm(normalized_shape=self.d_model)
        self.labelTransformerEncoder = TransformerEncoder(encoder_layer=labelEncoderLayer, 
                                                          num_layers=self.num_layer,
                                                          norm=labelEncoderNorm)
        self.hash = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=self.classes * self.nbit, out_channels=self.classes * self.nbit,kernel_size=1, groups=self.classes), 
            paddle.nn.Tanh())
        self.classify = paddle.nn.Linear(in_features=self.nbit,out_features=self.classes)

    def forward(self, label):
        self.batch_size = label.shape[0]

        index = paddle.arange(start=1, end=self.classes + 1).unsqueeze(axis=0)
        label_embedding = self.labelEmbedding(index)
        memory = self.labelTransformerEncoder(label_embedding.transpose(perm=[1, 0, 2]))
        
        prototype = self.hash(memory.transpose(perm=[1, 0, 2]).reshape([1, self.classes * self.nbit, 1, 1])).squeeze()
        prototype = prototype.squeeze().reshape([self.classes, self.nbit])
        
        code = paddle.matmul(x=label, y=prototype)
        pred = self.classify(code)
        return prototype, code, pred
