import os
import paddle
from paddle import nn
import numpy as np
from paddle.nn import functional as F
from paddle.io import DataLoader
from tqdm import tqdm
from datasets.dataset import Dataset
from config import opt
from models import *
from paddle.optimizer import Adamax
from utils import calc_map_k
from datasets.data_handler import load_data, load_pretrain_model
import time
from scipy.io import loadmat
# from mat4py import loadmat
import faulthandler
from models import agah, triplet_loss

paddle.device.set_device('gpu:7')
faulthandler.enable()

def train(**kwargs):
    # opt.parse(kwargs)

    images, tags, labels = load_data(opt.data_path, type=opt.dataset)
    print("load data success!")

    pretrain_model = None
    if opt.load_model_path:
        pretrain_model = None
    elif opt.pretrain_model_path:
        print(opt.pretrain_model_path)
        pretrain_model = load_pretrain_model(opt.pretrain_model_path)
        # pretrain_model = loadmat(opt.pretrain_model_path)
        print("load pretrain_model success")

    train_data = Dataset(opt, images, tags, labels)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    train_labels = train_data.get_labels().cuda()
    del train_data

    # valid or test data
    x_query_data = Dataset(opt, images, tags, labels, test='image.query')
    x_db_data = Dataset(opt, images, tags, labels, test='image.db')
    y_query_data = Dataset(opt, images, tags, labels, test='text.query')
    y_db_data = Dataset(opt, images, tags, labels, test='text.db')

    x_query_dataloader = DataLoader(x_query_data, batch_size=opt.batch_size, shuffle=False)
    x_db_dataloader = DataLoader(x_db_data, batch_size=opt.batch_size, shuffle=False)
    y_query_dataloader = DataLoader(y_query_data, batch_size=opt.batch_size, shuffle=False)
    y_db_dataloader = DataLoader(y_db_data, batch_size=opt.batch_size, shuffle=False)

    query_labels, db_labels = x_query_data.get_labels()
    query_labels = query_labels.cuda()
    db_labels = db_labels.cuda()
    
    for i, (ind, input_data) in enumerate(x_query_dataloader):
        input_data = paddle.to_tensor(input_data).cuda()
    
    model = agah.AGAH(opt.bit, opt.tag_dim, opt.num_label, opt.emb_dim,
                lambd=opt.lambd, pretrain_model=pretrain_model)

    load_model(model, opt.load_model_path)

    optimizer = {
        'img': Adamax(learning_rate = opt.lr, parameters = model.img_module.parameters(), weight_decay=0.0005),
        'txt': Adamax(learning_rate = opt.lr * 10, parameters = model.txt_module.parameters(), weight_decay=0.0005),
        'hash': Adamax(learning_rate = opt.lr * 10, parameters = model.hash_module.parameters(), weight_decay=0.0005),
        'classifier': Adamax(learning_rate = opt.lr * 10, parameters = model.classifier.parameters(), weight_decay=0.0005)
    }
    
    # Adamax(learning_rate = opt.lr * 10, parameters = [{'params': model.img_module.parameters(), 'learning_rate': opt.lr},
    #     {'params': model.txt_module.parameters()},
    #     {'params': model.hash_module.parameters()},
    #     {'params': model.classifier.parameters()}], weight_decay=0.0005)

    optimizer_dis = {
        'img': Adamax(learning_rate = opt.lr * 10, parameters = model.img_discriminator.parameters(), beta1 = 0.5, beta2 = 0.9, weight_decay=0.0001),
        'txt': Adamax(learning_rate = opt.lr * 10, parameters = model.txt_discriminator.parameters(), beta1 = 0.5, beta2 = 0.9, weight_decay=0.0001)
    }

    criterion_tri_cos = triplet_loss.TripletAllLoss(dis_metric='cos', reduction='sum')
    criterion_bce = nn.BCELoss(reduction='sum')

    loss = []

    max_mapi2t = 0.
    max_mapt2i = 0.

    FEATURE_I = paddle.randn([opt.training_size, opt.emb_dim]).cuda()
    FEATURE_T = paddle.randn([opt.training_size, opt.emb_dim]).cuda()

    U = paddle.randn([opt.training_size, opt.bit]).cuda()
    V = paddle.randn([opt.training_size, opt.bit]).cuda()


    FEATURE_MAP = paddle.randn([opt.num_label, opt.emb_dim]).cuda()
    CODE_MAP = paddle.sign(paddle.randn([opt.num_label, opt.bit])).cuda()

    mapt2i_list = []
    mapi2t_list = []
    train_times = []

    for epoch in range(opt.max_epoch):
        t1 = time.time()
        for i, (ind, x, y, l) in tqdm(enumerate(train_dataloader)):
            imgs = x.cuda()
            tags = y.cuda()
            labels = l.cuda()

            batch_size = len(ind)

            h_x, h_y, f_x, f_y, x_class, y_class = model(imgs, tags, FEATURE_MAP)
            FEATURE_I[ind, :] = f_x.detach()
            FEATURE_T[ind, :] = f_y.detach()
            U[ind, :] = h_x.detach()
            V[ind, :] = h_y.detach()

            #####
            # train txt discriminator
            #####
            D_txt_real = model.dis_txt(f_y.detach())
            D_txt_real = -D_txt_real.mean()
            optimizer_dis['txt'].clear_grad()
            D_txt_real.backward()

            # train with fake
            D_txt_fake = model.dis_txt(f_x.detach())
            D_txt_fake = D_txt_fake.mean()
            D_txt_fake.backward()

            # train with gradient penalty
            alpha = paddle.rand([batch_size, opt.emb_dim]).cuda()
            interpolates = alpha * f_y.detach() + (1 - alpha) * f_x.detach()
            interpolates.stop_gradient = False
            disc_interpolates = model.dis_txt(interpolates)
            gradients = paddle.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=paddle.ones(disc_interpolates.shape).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = paddle.reshape(gradients, [gradients.shape[0], -1])
            # 10 is gradient penalty hyperparameter
            txt_gradient_penalty = ((paddle.norm(gradients, 2, axis=1) - 1) ** 2).mean() * 10
            txt_gradient_penalty.backward()

            loss_D_txt = D_txt_real - D_txt_fake
            optimizer_dis['txt'].step()

            #####
            # train img discriminator
            #####
            D_img_real = model.dis_img(f_x.detach())
            D_img_real = -D_img_real.mean()
            optimizer_dis['img'].clear_grad()
            D_img_real.backward()

            # train with fake
            D_img_fake = model.dis_img(f_y.detach())
            D_img_fake = D_img_fake.mean()
            D_img_fake.backward()

            # train with gradient penalty
            alpha = paddle.rand([batch_size, opt.emb_dim]).cuda()
            interpolates = alpha * f_x.detach() + (1 - alpha) * f_y.detach()
            interpolates.stop_gradient = False
            disc_interpolates = model.dis_img(interpolates)
            gradients = paddle.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=paddle.ones(disc_interpolates.shape).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = paddle.reshape(gradients, [gradients.shape[0], -1])
            # 10 is gradient penalty hyperparameter
            img_gradient_penalty = ((paddle.norm(gradients, 2, axis=1) - 1) ** 2).mean() * 10
            img_gradient_penalty.backward()

            loss_D_img = D_img_real - D_img_fake
            optimizer_dis['img'].step()

            #####
            # train generators
            #####
            # update img network (to generate txt features)
            domain_output = model.dis_txt(f_x)
            loss_G_txt = -domain_output.mean()

            # update txt network (to generate img features)
            domain_output = model.dis_img(f_y)
            loss_G_img = -domain_output.mean()

            loss_adver = loss_G_txt + loss_G_img

            loss1 = criterion_tri_cos(h_x, labels, target=h_y, margin=opt.margin)
            loss2 = criterion_tri_cos(h_y, labels, target=h_x, margin=opt.margin)

            theta1 = F.cosine_similarity(paddle.abs(h_x), paddle.ones_like(h_x)).cuda()
            theta2 = F.cosine_similarity(paddle.abs(h_y), paddle.ones_like(h_y)).cuda()
            loss3 = paddle.sum(1 / (1 + paddle.exp(theta1))) + paddle.sum(1 / (1 + paddle.exp(theta2)))

            loss_class = criterion_bce(x_class, labels) + criterion_bce(y_class, labels)

            theta_code_x = paddle.matmul(h_x, paddle.transpose(CODE_MAP, [1, 0]))  # size: (batch, num_label)
            theta_code_y = paddle.matmul(h_y, paddle.transpose(CODE_MAP, [1, 0]))
            loss_code_map = paddle.sum(paddle.pow(theta_code_x - opt.bit * (labels * 2 - 1), 2)) + \
                            paddle.sum(paddle.pow(theta_code_y - opt.bit * (labels * 2 - 1), 2))

            loss_quant = paddle.sum(paddle.pow(h_x - paddle.sign(h_x), 2)) + paddle.sum(paddle.pow(h_y - paddle.sign(h_y), 2))

            # err = loss1 + loss2 + loss3 + 0.5 * loss_class + 0.5 * (loss_f1 + loss_f2)
            err = loss1 + loss2 + opt.alpha * loss3 + opt.beta * loss_class + opt.gamma * loss_code_map + \
                  opt.eta * loss_quant + opt.mu * loss_adver

            optimizer['img'].clear_grad()
            optimizer['txt'].clear_grad()
            optimizer['hash'].clear_grad()
            optimizer['classifier'].clear_grad()
            # optimizer.clear_grad()
            err.backward()
            optimizer['img'].step()
            optimizer['txt'].step()
            optimizer['hash'].step()
            optimizer['classifier'].step()
            # optimizer.step()

            loss.append(err)

        CODE_MAP = update_code_map(U, V, CODE_MAP, train_labels)
        FEATURE_MAP = update_feature_map(FEATURE_I, FEATURE_T, train_labels)

        print('...epoch: %3d, loss: %3.3f' % (epoch + 1, loss[-1]))
        delta_t = time.time() - t1

        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i = valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
                                   query_labels, db_labels, FEATURE_MAP)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            train_times.append(delta_t)

            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                save_model(model)
                path = './checkpoint/AGAH_' + opt.dataset + '_' + str(opt.bit)
                paddle.save(FEATURE_MAP, os.path.join(path, '_feature_map.pdparams'))

        if epoch % 100 == 0:
            # for params in optimizer.param_groups:
            #     params['lr'] = max(params['lr'] * 0.6, 1e-6)
            optimizer['img'].set_lr(max(optimizer['img'].get_lr() * 0.6, 1e-6))
            optimizer['txt'].set_lr(max(optimizer['txt'].get_lr() * 0.6, 1e-6))
            optimizer['hash'].set_lr(max(optimizer['hash'].get_lr() * 0.6, 1e-6))
            optimizer['classifier'].set_lr(max(optimizer['classifier'].get_lr() * 0.6, 1e-6))

    if opt.valid:
        save_model(model)

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i = valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
                               query_labels, db_labels, FEATURE_MAP)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))

def update_code_map(U, V, M, L):
    CODE_MAP = M
    U = paddle.sign(U)
    V = paddle.sign(V)
    S = paddle.to_tensor(paddle.eye(opt.num_label) * 2 - 1)

    Q = 2 * opt.bit * (paddle.matmul(paddle.transpose(L, [1, 0]), U + V) + paddle.matmul(S, M))

    for k in range(opt.bit):
        ind = np.setdiff1d(np.arange(0, opt.bit), k)
        ind = paddle.to_tensor(ind, dtype='int32')
        term1 = paddle.matmul(paddle.matmul(paddle.index_select(CODE_MAP, ind, axis=1), paddle.transpose(paddle.index_select(U, ind, axis=1), perm=[1, 0])), (U[:, k].unsqueeze(-1))).squeeze()
        term2 = paddle.matmul(paddle.matmul(paddle.index_select(CODE_MAP, ind, axis=1), paddle.transpose(paddle.index_select(V, ind, axis=1), perm=[1, 0])), (V[:, k].unsqueeze(-1))).squeeze()
        term3 = paddle.matmul(paddle.matmul(paddle.index_select(CODE_MAP, ind, axis=1), paddle.transpose(paddle.index_select(M, ind, axis=1), perm=[1, 0])), (M[:, k].unsqueeze(-1))).squeeze()
        CODE_MAP[:, k] = paddle.sign(Q[:, k] - 2 * (term1 + term2 + term3))

    return CODE_MAP


def update_feature_map(FEAT_I, FEAT_T, L, mode='average'):
    if mode == 'average':
        feature_map_I = paddle.matmul(paddle.transpose(L, [1, 0]), FEAT_I) / paddle.unsqueeze(L.sum(axis=0), axis=-1)
        feature_map_T = paddle.matmul(paddle.transpose(L, [1, 0]), FEAT_T) / paddle.unsqueeze(L.sum(axis=0), axis=-1)
    else:
        assert mode == 'max'
        feature_map_I = (paddle.unsqueeze(paddle.transpose(L, [1, 0]), axis=-1) * FEAT_I).max(axis=1)[0]
        feature_map_T = (paddle.unsqueeze(paddle.transpose(L, [1, 0]), axis=-1) * FEAT_T).max(axis=1)[0]

    FEATURE_MAP = (feature_map_T + feature_map_I) / 2
    # normalization
    FEATURE_MAP = FEATURE_MAP / paddle.sqrt(paddle.sum(FEATURE_MAP ** 2, axis=-1, keepdim=True))
    return FEATURE_MAP


def valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
          query_labels, db_labels, FEATURE_MAP):
    model.eval()

    qBX = generate_img_code(model, x_query_dataloader, opt.query_size, FEATURE_MAP)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size, FEATURE_MAP)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size, FEATURE_MAP)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size, FEATURE_MAP)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels, 5000)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels, 5000)

    model.train()
    return mapi2t.item(), mapt2i.item()


def test(**kwargs):
    opt.parse(kwargs)

    if opt.load_model_path:
        pretrain_model = None
    elif opt.pretrain_model_path:
        pretrain_model = load_pretrain_model(opt.pretrain_model_path)

    model = agah.AGAH(opt.bit, opt.tag_dim, opt.num_label, opt.emb_dim,
                lambd=opt.lambd, pretrain_model=pretrain_model)

    path = './checkpoint/AGAH_' + opt.dataset + '_' + str(opt.bit)
    load_model(model, path)
    FEATURE_MAP = paddle.load(os.path.join(path, '_feature_map.pdparams')).cuda()

    model.eval()

    images, tags, labels = load_data(opt.data_path, opt.dataset)

    x_query_data = Dataset(opt, images, tags, labels, test='image.query')
    x_db_data = Dataset(opt, images, tags, labels, test='image.db')
    y_query_data = Dataset(opt, images, tags, labels, test='text.query')
    y_db_data = Dataset(opt, images, tags, labels, test='text.db')

    x_query_dataloader = DataLoader(x_query_data, opt.batch_size, shuffle=False)
    x_db_dataloader = DataLoader(x_db_data, opt.batch_size, shuffle=False)
    y_query_dataloader = DataLoader(y_query_data, opt.batch_size, shuffle=False)
    y_db_dataloader = DataLoader(y_db_data, opt.batch_size, shuffle=False)

    qBX = generate_img_code(model, x_query_dataloader, opt.query_size, FEATURE_MAP)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size, FEATURE_MAP)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size, FEATURE_MAP)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size, FEATURE_MAP)

    query_labels, db_labels = x_query_data.get_labels()
    query_labels = query_labels.cuda()
    db_labels = db_labels.cuda()

    # p_i2t, r_i2t = pr_curve(qBX, rBY, query_labels, db_labels)
    # p_t2i, r_t2i = pr_curve(qBY, rBX, query_labels, db_labels)

    K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # pk_i2t = p_topK(qBX, rBY, query_labels, db_labels, K)
    # pk_t2i = p_topK(qBY, rBX, query_labels, db_labels, K)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels, 5000)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels, 5000)
    print('...test MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))


def generate_img_code(model, test_dataloader, num, FEATURE_MAP):
    B = paddle.zeros([num, opt.bit])

    for i, (ind, input_data) in enumerate(test_dataloader):
        input_data = paddle.to_tensor(input_data).cuda()
        b = model.generate_img_code(input_data, FEATURE_MAP)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.detach()

    B = paddle.sign(B)
    return B


def generate_txt_code(model, test_dataloader, num, FEATURE_MAP):
    B = paddle.zeros([num, opt.bit])

    for i, (ind, input_data) in enumerate(test_dataloader):
        input_data = paddle.to_tensor(input_data).cuda()
        b = model.generate_txt_code(input_data, FEATURE_MAP)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.detach()

    B = paddle.sign(B)
    return B


def calc_loss(loss):
    l = 0.
    for v in loss.values():
        l += v[-1]
    return l


def avoid_inf(x):
    return paddle.log(1.0 + paddle.exp(-paddle.abs(x))) + paddle.maximum(paddle.zeros_like(x), x)


def load_model(model, path):
    if path is not None:
        model.set_state_dict(paddle.load(path + '.pdparams'))


def save_model(model):
    path = './checkpoint/AGAH_' + opt.dataset + '_' + str(opt.bit) + '.pdparams'
    paddle.save(model.state_dict(), path)


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''========================::HELP::=========================
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example:
            python {0} train --lr=0.01
            python {0} help
    avaiable args (default value):'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__') and str(k) != 'parse':
            print('            {0}: {1}'.format(k, v))
    print('========================::HELP::=========================')


if __name__ == '__main__':
    import fire
    fire.Fire()
