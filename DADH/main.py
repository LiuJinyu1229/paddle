import os
from paddle import autograd
from paddle.io import DataLoader
from config import opt
from models.dis_model import DIS
from models.gen_model import GEN
from triplet_loss import *
from paddle.optimizer import Adam
from utils import calc_map
import time
import pickle
from datasets.data_handler import load_data
from datasets.dataset import Dataset
from tqdm import tqdm
import argparse

# from dset import MY_DATASET
paddle.device.set_device('gpu:4')

def train(**kwargs):
    opt.parse(kwargs)
    opt.beta = opt.beta + 0.1

    images, tags, labels = load_data(opt.data_path, type=opt.dataset)
    print("load data success!")
    train_data = Dataset(opt, images, tags, labels)
    print("Dataset")
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    print("DataLoad")
    L = train_data.get_labels()
    L = L.cuda()

    # train_data = MY_DATASET(opt=opt)
    # train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    # L = train_data.labels
    # L = L.cuda()
    # opt.training_size = L.shape[0]
    # opt.num_label = L.shape[1]
    # opt.text_dim = train_data.txt.shape[1]

    # test
    i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, batch_size=opt.batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, batch_size=opt.batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, batch_size=opt.batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, batch_size=opt.batch_size, shuffle=False)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.cuda()
    db_labels = db_labels.cuda()

    pretrain_model = None

    generator = GEN(opt.dropout, opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, pretrain_model=pretrain_model)
    print("load GEN success!")
    discriminator = DIS(opt.hidden_dim//4, opt.hidden_dim//8, opt.bit)
    print("load DIS success!")

    optimizer = Adam(parameters=generator.image_module.parameters() + generator.text_module.parameters() + generator.hash_module.parameters(), learning_rate=opt.lr, weight_decay=0.0005)

    optimizer_dis = {
        'feature': Adam(parameters=discriminator.feature_dis.parameters(), learning_rate=opt.lr, beta1=0.5, beta2=0.9, weight_decay=0.0001),
        'hash': Adam(parameters=discriminator.hash_dis.parameters(), learning_rate=opt.lr, beta1=0.5, beta2=0.9, weight_decay=0.0001)
    }

    tri_loss = TripletLoss(opt, reduction='sum')

    loss = []

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_average = 0.

    mapt2i_list = []
    mapi2t_list = []
    train_times = []

    B_i = paddle.randn([opt.training_size, opt.bit]).sign().cuda()
    B_t = B_i
    H_i = paddle.zeros([opt.training_size, opt.bit]).cuda()
    H_t = paddle.zeros([opt.training_size, opt.bit]).cuda()

    print('...training procedure start')
    for epoch in range(opt.max_epoch):
        t1 = time.time()
        e_loss = 0
        # for i, (img, txt, label, ind) in tqdm(enumerate(train_dataloader)):
        for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
            if paddle.is_compiled_with_cuda():
                imgs = img.cuda()
                txt = txt.cuda()
                labels = label.cuda()
            else:
                imgs = img
                txt = txt
                labels = label
            # print("aaaa")

            batch_size = len(ind)
            h_i, h_t, f_i, f_t = generator(imgs, txt)
            H_i[ind, :] = h_i.numpy()
            H_t[ind, :] = h_t.numpy()
            h_t_detach = generator.generate_txt_code(txt)

            #####
            # train feature discriminator
            #####
            D_real_feature = discriminator.dis_feature(f_i.detach())
            D_real_feature = -opt.gamma * paddle.log(paddle.nn.functional.sigmoid(D_real_feature)).mean()
            # D_real_feature = -D_real_feature.mean()
            optimizer_dis['feature'].clear_grad()
            D_real_feature.backward()

            # train with fake
            D_fake_feature = discriminator.dis_feature(f_t.detach())
            D_fake_feature = -opt.gamma * paddle.log(paddle.ones([batch_size]).cuda() - paddle.nn.functional.sigmoid(D_fake_feature)).mean()
            # D_fake_feature = D_fake_feature.mean()
            D_fake_feature.backward()

            # train with gradient penalty
            alpha = paddle.rand([batch_size, opt.hidden_dim//4]).cuda()
            interpolates = alpha * f_i.detach() + (1 - alpha) * f_t.detach()
            interpolates.stop_gradient = False
            disc_interpolates = discriminator.dis_feature(interpolates)
            gradients = paddle.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=paddle.ones(disc_interpolates.shape).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.reshape([gradients.shape[0], -1])
            # 10 is gradient penalty hyperparameter
            feature_gradient_penalty = ((paddle.norm(gradients, 2, axis=1) - 1) ** 2).mean() * 10
            feature_gradient_penalty.backward()

            optimizer_dis['feature'].step()

            #####
            # train hash discriminator
            #####
            D_real_hash = discriminator.dis_hash(h_i.detach())
            D_real_hash = -opt.gamma * paddle.log(paddle.nn.functional.sigmoid(D_real_hash)).mean()
            optimizer_dis['hash'].clear_grad()
            D_real_hash.backward()

            # train with fake
            D_fake_hash = discriminator.dis_hash(h_t.detach())
            D_fake_hash = -opt.gamma * paddle.log(paddle.ones([batch_size]).cuda() - paddle.nn.functional.sigmoid(D_fake_hash)).mean()
            D_fake_hash.backward()

            # train with gradient penalty
            alpha = paddle.rand([batch_size, opt.bit]).cuda()
            interpolates = alpha * h_i.detach() + (1 - alpha) * h_t.detach()
            interpolates.stop_gradient = False
            disc_interpolates = discriminator.dis_hash(interpolates)
            gradients = paddle.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=paddle.ones(disc_interpolates.shape).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.reshape([gradients.shape[0], -1])

            hash_gradient_penalty = ((paddle.norm(gradients, 2, axis=1) - 1) ** 2).mean() * 10
            hash_gradient_penalty.backward()

            optimizer_dis['hash'].step()

            loss_G_txt_feature = -paddle.log(paddle.nn.functional.sigmoid(discriminator.dis_feature(f_t))).mean()
            loss_adver_feature = loss_G_txt_feature

            loss_G_txt_hash = -paddle.log(paddle.nn.functional.sigmoid(discriminator.dis_hash(h_t_detach))).mean()
            loss_adver_hash = loss_G_txt_hash

            tri_i2t = tri_loss(h_i, labels, target=h_t, margin=opt.margin)
            tri_t2i = tri_loss(h_t, labels, target=h_i, margin=opt.margin)
            weighted_cos_tri = tri_i2t + tri_t2i
            # print(type(ind))
            # # ind = paddle.to_tensor(ind)
            # ind = ind.numpy().tolist()
            # print(ind)
            i_ql = 0
            t_ql = 0
            for i in ind:
                i_ql += paddle.sum((B_i[i, :] - h_i) ** 2)
                t_ql += paddle.sum((B_i[i, :] - h_t) ** 2)
            # i_ql = paddle.sum((B_i[ind, :] - h_i) ** 2)
            # t_ql = paddle.sum((B_i[ind, :] - h_t) ** 2)
            loss_quant = i_ql + t_ql
            err = opt.alpha * weighted_cos_tri + \
                opt.beta * loss_quant + opt.gamma * (loss_adver_feature + loss_adver_hash)

            optimizer.clear_grad()
            err.backward()
            optimizer.step()

            e_loss = err + e_loss

        P_i = paddle.inverse(
                        L.t() @ L + opt.lamb * paddle.eye(opt.num_label)) @ L.t() @ B_i

        B_i = (L @ P_i + 0.5 * opt.mu * (H_i + H_t)).sign()
        loss.append(e_loss.item())
        print('...epoch: %3d, loss: %3.3f' % (epoch + 1, loss[-1]))
        delta_t = time.time() - t1

        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i = valid(generator, i_query_dataloader, i_db_dataloader, t_query_dataloader, t_db_dataloader,
                                query_labels, db_labels)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            train_times.append(delta_t)

            if 0.5 * (mapi2t + mapt2i) > max_average:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                max_average = 0.5 * (mapi2t + mapt2i)
                save_model(generator)

        if epoch % 100 == 0:
            current_lr = optimizer.get_lr()
            optimizer.set_lr(max(current_lr * 0.8, 1e-7))

    if not opt.valid:
        save_model(generator)

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i = valid(generator, i_query_dataloader, i_db_dataloader, t_query_dataloader, t_db_dataloader,
                            query_labels, db_labels)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))


def valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
          query_labels, db_labels):
    model.eval()

    qBX = generate_img_code(model, x_query_dataloader, opt.query_size)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size)

    mapi2t = calc_map(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map(qBY, rBX, query_labels, db_labels)

    model.train()
    return mapi2t, mapt2i

def test(**kwargs):
    print("start to test...")
    opt.parse(kwargs)

    # pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    print("start to load model...")
    pretrain_model = None
    generator = GEN(opt.dropout, opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, pretrain_model=pretrain_model)
    path = 'checkpoint/DADH_' + opt.dataset + '_' + str(opt.bit)
    load_model(generator, path)
    print("load model success!")

    generator.eval()

    # images, tags, labels = load_data(opt.data_path, opt.dataset)
    #
    # i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    # i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    # t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    # t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    # test
    images, tags, labels = load_data(opt.data_path, opt.dataset)

    i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, batch_size=opt.batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, batch_size=opt.batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, batch_size=opt.batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, batch_size=opt.batch_size, shuffle=False)

    # opt.query_size = query_labels.labels.shape[0]
    # opt.db_size = db_labels.labels.shape[0]

    # i_query_dataloader = paddle.io.DataLoader(i_query_data, batch_size=opt.batch_size, shuffle=False)
    # i_db_dataloader = paddle.io.DataLoader(i_db_data, batch_size=opt.batch_size, shuffle=False)
    # t_query_dataloader = paddle.io.DataLoader(t_query_data, batch_size=opt.batch_size, shuffle=False)
    # t_db_dataloader = paddle.io.DataLoader(t_db_data, batch_size=opt.batch_size, shuffle=False)

    qBX = generate_img_code(generator, i_query_dataloader, opt.query_size)
    qBY = generate_txt_code(generator, t_query_dataloader, opt.query_size)
    rBX = generate_img_code(generator, i_db_dataloader, opt.db_size)
    rBY = generate_txt_code(generator, t_db_dataloader, opt.db_size)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.cuda()
    db_labels = db_labels.cuda()

    mapi2t = calc_map(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map(qBY, rBX, query_labels, db_labels)
    print('...test MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
    print("end test...")


def generate_img_code(model, test_dataloader, num):
    B = paddle.zeros([num, opt.bit])
    for i, input_data in tqdm(enumerate(test_dataloader)):
        input_data = paddle.to_tensor(input_data)
        b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b

    B = paddle.sign(B)
    return B


def generate_txt_code(model, test_dataloader, num):
    B = paddle.zeros([num, opt.bit])

    for i, input_data in tqdm(enumerate(test_dataloader)):
        input_data = paddle.to_tensor(input_data)
        b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b

    B = paddle.sign(B)
    return B


def load_model(model, path):
    if path is not None:
        path = path + '_' + model.module_name + '.pdparams'
        model.set_state_dict(paddle.load(path))


def save_model(model):
    path = 'checkpoint/DADH_' + opt.dataset + '_' + str(opt.bit)+ '_' + model.module_name + '.pdparams'
    paddle.save(model.state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--dataset', type=str, default='mir', help='Dataset')
    args = parser.parse_args()
    if args.train:
        train(flag='mir')
    else:
        test(flag='mir')
