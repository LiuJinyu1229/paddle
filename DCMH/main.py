from config import opt
from data_handler import *
import numpy as np
import paddle
from paddle import nn
from paddle.optimizer import SGD
from tqdm import tqdm
from models import ImgModule, TxtModule
from utils import calc_map_k

paddle.device.set_device('gpu:6')

def train(**kwargs):
    # opt.parse(kwargs)

    print("start loading data")
    images, tags, labels = load_data(opt.data_path)
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    y_dim = tags.shape[1]

    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit, pretrain_model)
    txt_model = TxtModule(y_dim, opt.bit)

    train_L = paddle.to_tensor(L['train'], dtype='float32')
    train_x = paddle.to_tensor(X['train'], dtype='float32')
    train_y = paddle.to_tensor(Y['train'], dtype='float32')

    query_L = paddle.to_tensor(L['query'], dtype='float32')
    query_x = paddle.to_tensor(X['query'], dtype='float32')
    query_y = paddle.to_tensor(Y['query'], dtype='float32')

    retrieval_L = paddle.to_tensor(L['retrieval'], dtype='float32')
    retrieval_x = paddle.to_tensor(X['retrieval'], dtype='float32')
    retrieval_y = paddle.to_tensor(Y['retrieval'], dtype='float32')

    num_train = train_x.shape[0]

    F_buffer = paddle.randn([num_train, opt.bit])
    G_buffer = paddle.randn([num_train, opt.bit])

    train_L = paddle.to_tensor(train_L, dtype='float32')
    F_buffer = paddle.to_tensor(F_buffer, dtype='float32')
    G_buffer = paddle.to_tensor(G_buffer, dtype='float32')

    Sim = calc_neighbor(train_L, train_L)
    B = paddle.sign(F_buffer + G_buffer)

    batch_size = opt.batch_size

    lr = opt.lr
    optimizer_img = SGD(parameters=img_model.parameters(), learning_rate=lr)
    optimizer_txt = SGD(parameters=txt_model.parameters(), learning_rate=lr)

    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': []
    }

    ones = paddle.ones([batch_size, 1])
    ones_ = paddle.ones([num_train - batch_size, 1])
    unupdated_size = num_train - batch_size

    max_mapi2t = max_mapt2i = 0.

    for epoch in range(opt.max_epoch):
        # train image net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = paddle.to_tensor(train_L[ind, :])
            image = paddle.to_tensor(train_x[ind].astype('float32'))
            ones = paddle.to_tensor(ones, dtype='float32')
            ones_ = paddle.to_tensor(ones_, dtype='float32')

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            cur_f = img_model(image)  # cur_f: (batch_size, bit)
            F_buffer[ind, :] = cur_f.numpy()
            F = paddle.to_tensor(F_buffer)
            G = paddle.to_tensor(G_buffer)

            theta_x = 1.0 / 2 * paddle.matmul(cur_f, G.t())
            logloss_x = -paddle.sum(S * theta_x - paddle.log(1.0 + paddle.exp(theta_x)))
            quantization_x = paddle.sum(paddle.pow(B[ind, :] - cur_f, 2))
            balance_x = paddle.sum(paddle.pow(paddle.matmul(cur_f.t(), ones) + paddle.matmul(F[unupdated_ind].t(), ones_), 2))
            loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x
            loss_x /= (batch_size * num_train)

            optimizer_img.clear_grad()
            loss_x.backward()
            optimizer_img.step()

        # train txt net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = paddle.to_tensor(train_L[ind, :])
            text = paddle.unsqueeze(paddle.unsqueeze(train_y[ind, :], 1), -1).astype('float32')
            text = paddle.to_tensor(text)

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            cur_g = txt_model(text)  # cur_f: (batch_size, bit)
            G_buffer[ind, :] = cur_g.numpy()
            F = paddle.to_tensor(F_buffer)
            G = paddle.to_tensor(G_buffer)

            # calculate loss
            # theta_y: (batch_size, num_train)
            theta_y = 1.0 / 2 * paddle.matmul(cur_g, F.t())
            logloss_y = -paddle.sum(S * theta_y - paddle.log(1.0 + paddle.exp(theta_y)))
            quantization_y = paddle.sum(paddle.pow(B[ind, :] - cur_g, 2))
            balance_y = paddle.sum(paddle.pow(paddle.matmul(cur_g.t(), ones) + paddle.matmul(G[unupdated_ind].t(), ones_), 2))
            loss_y = logloss_y + opt.gamma * quantization_y + opt.eta * balance_y
            loss_y /= (num_train * batch_size)

            optimizer_txt.clear_grad()
            loss_y.backward()
            optimizer_txt.step()

        # update B
        B = paddle.sign(F_buffer + G_buffer)

        # calculate total loss
        loss = calc_loss(B, F, G, paddle.to_tensor(Sim), opt.gamma, opt.eta)

        print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.numpy(), lr))
        result['loss'].append(float(loss.numpy()))

        if opt.valid:
            mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                                query_L, retrieval_L)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                file_path = './checkpoint/DCMH_%s_%d.pdparams' % (opt.dataset, opt.bit)
                obj = {
                    'img':img_model.state_dict(),
                    'txt': txt_model.state_dict(),
                }
                paddle.save(obj, file_path)

        lr = learning_rate[epoch + 1]

        # set learning rate
        optimizer_img.set_lr(lr)
        optimizer_txt.set_lr(lr)

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        result['mapi2t'] = max_mapi2t
        result['mapt2i'] = max_mapt2i
    else:
        mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                               query_L, retrieval_L)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        result['mapi2t'] = mapi2t
        result['mapt2i'] = mapt2i

def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def test(**kwargs):
    opt.parse(kwargs)

    images, tags, labels = load_data(opt.data_path)
    y_dim = tags.shape[1]

    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    file_path = './checkpoint/DCMH_%s_%d.pdparams' % (opt.dataset, opt.bit)
    obj = paddle.load(file_path)
    img_model.set_state_dict(obj['img'])
    txt_model.set_state_dict(obj['txt'])

    query_L = paddle.to_tensor(L['query'], dtype='float32')
    query_x = paddle.to_tensor(X['query'], dtype='float32')
    query_y = paddle.to_tensor(Y['query'], dtype='float32')

    retrieval_L = paddle.to_tensor(L['retrieval'], dtype='float32')
    retrieval_x = paddle.to_tensor(X['retrieval'], dtype='float32')
    retrieval_y = paddle.to_tensor(Y['retrieval'], dtype='float32')

    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))


def split_data(images, tags, labels):
    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    if opt.use_gpu:
        Sim = (paddle.matmul(label1, paddle.transpose(label2, [1, 0])) > 0).astype('float32')
    else:
        Sim = (paddle.matmul(label1, paddle.transpose(label2, [1, 0])) > 0).astype('float32')
    return Sim


def calc_loss(B, F, G, Sim, gamma, eta):
    theta = paddle.matmul(F, paddle.transpose(G, [1, 0])) / 2
    term1 = paddle.sum(paddle.log(1 + paddle.exp(theta)) - Sim * theta)
    term2 = paddle.sum(paddle.pow(B - F, 2) + paddle.pow(B - G, 2))
    term3 = paddle.sum(paddle.pow(paddle.sum(F, axis=0), 2) + paddle.pow(paddle.sum(G, axis=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = paddle.zeros([num_data, bit], dtype='float32')
    B = paddle.to_tensor(B, dtype='float32')
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = paddle.to_tensor(X[ind].astype('float32'))
        cur_f = img_model(image)
        B[ind, :] = cur_f.numpy()
    B = paddle.sign(B)
    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = paddle.zeros([num_data, bit], dtype='float32')
    B = paddle.to_tensor(B, dtype='float32')
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = paddle.unsqueeze(paddle.unsqueeze(paddle.to_tensor(Y[ind].astype('float32')), 1), -1)
        cur_g = txt_model(text)
        B[ind, :] = cur_g.numpy()
    B = paddle.sign(B)
    return B

def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    import fire
    fire.Fire()
