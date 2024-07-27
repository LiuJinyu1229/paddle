import argparse
import os

import numpy as np
import paddle
import h5py

from copy import deepcopy
from datasets import ImgFile
from label_module import LabelModule
from autoencoder import AutoEncoderGcnModule
from fusion_module import *
from evaluate import *
from losses import *
import utils

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='flickr')
    parser.add_argument('--batch_size', type=int, default=250, help='number of images in a batch')
    parser.add_argument('--bit', type=int, default=32, help='length of hash codes')
    parser.add_argument('--Epoch_num', type=int, default=2, help='num of Epochs')
    parser.add_argument('--times', type=int, default=1, help='num of times')
    parser.add_argument('--nc', type=int, default=3000, help='complete pairs')
    parser.add_argument('--n1u', type=int, default=1000, help='incomplete pairs with only images')
    parser.add_argument('--n2u', type=int, default=1000, help='incomplete pairs with only texts')
    parser.add_argument('--gamma', type=float, default=10, help='balance the importance of image/text')
    parser.add_argument('--lamda', type=float, default=10, help='lamda')
    parser.add_argument('--a', type=float, default=0.3, help='a')
    parser.add_argument('--alpha', type=int, default=14, help='alpha')
    parser.add_argument('--beta', type=float, default=0.0000001, help='beta')
    parser.add_argument('--p1', type=float, default=0.4, help='node itself')
    parser.add_argument('--p2', type=float, default=0.02, help='node itself')
    parser.add_argument('--c', type=float, default=0.8, help='GCN propogation')
    parser.add_argument('--train', action='store_true', help='Training mode')

    return parser.parse_args()


def generate_train_ds(images, texts, labels):
    # import ipdb
    # ipdb.set_trace()
    index = np.arange(labels.shape[0])
    np.random.shuffle(index)
    index1 = index[0:args.nc]
    index2 = index[args.nc:(args.nc + args.n1u)]
    index3 = index[(args.nc + args.n1u):(args.nc + args.n1u + args.n2u)]
    index1 = paddle.to_tensor(index1)
    index2 = paddle.to_tensor(index2)
    index3 = paddle.to_tensor(index3)
    image1 = paddle.stack([images[i] for i in index1])
    image1 = utils.normalize(image1)
    text1 = paddle.stack([texts[i] for i in index1])
    label1 = paddle.stack([labels[i] for i in index1])
    image2 = paddle.stack([images[i] for i in index2])
    image2 = utils.normalize(image2)
    label2 = paddle.stack([labels[i] for i in index2])
    text3 = paddle.stack([texts[i] for i in index3])
    label3 = paddle.stack([labels[i] for i in index3])

    # The mean values of existing image and text features are used to fill in the missing parts
    mean_text = np.mean(np.concatenate([text1, text3], axis=0), axis=0)
    mean_text = mean_text.reshape(1, len(mean_text))
    text2 = np.tile(mean_text, (args.n1u, 1))
    mean_image = np.mean(np.concatenate([image1, image2], axis=0), axis=0)
    mean_image = mean_image.reshape(1, len(mean_image))
    mean_image = paddle.to_tensor(mean_image)
    image1 = image1 - mean_image
    image2 = image2 - mean_image
    image3 = np.zeros((args.n2u, images.shape[1])).astype(np.float32)

    # All the features after completion
    images = np.concatenate([image1, image2, image3], axis=0)
    texts = np.concatenate([text1, text2, text3], axis=0)
    labels = np.concatenate([label1, label2, label3], axis=0)
    index = np.arange(labels.shape[0])
    np.random.shuffle(index)
    images = images[index]
    texts = texts[index]
    labels = labels[index]

    M1 = np.expand_dims((index < (args.nc + args.n1u)).astype(np.float32), axis=1) # only images
    M2 = np.expand_dims((index < args.nc).astype(np.float32) + (index >= (args.nc + args.n1u)).astype(np.float32), axis=1) # only texts

    datasets = ImgFile(images, texts, labels, M1, M2)
    data_loader = paddle.io.DataLoader(dataset=datasets, batch_size=args.batch_size, shuffle=True)
    return data_loader, paddle.to_tensor(labels).astype('float32'), labels.shape[1], texts.shape[1], labels.shape[0], mean_image, mean_text

def generate_test_database_ds(images, texts, labels):
    images = utils.normalize(images)
    images = images - mean_image
    datasets = ImgFile(images, texts, labels, np.ones([labels.shape[0], 1]), np.ones([labels.shape[0], 1]))
    data_loader = paddle.io.DataLoader(dataset=datasets, batch_size=args.batch_size, shuffle=False)
    return data_loader, paddle.to_tensor(labels).astype('float32')


def evaluate():
    # Set the model to testing mode
    label_model.eval()
    autoencoder_gcn_model.eval()
    fusion_model.eval()

    database_codes = []
    for image, text, _, _, _ in database_loader:
        image = image.cuda()
        text = text.cuda()
        _, h_fusion = fusion_model(paddle.concat((args.gamma * image, text), 1))
        codes = paddle.sign(h_fusion)
        database_codes.append(codes.numpy())
    database_codes = np.concatenate(database_codes)

    test_codes = []
    for image, text, _, _, _ in test_loader:
        image = image.cuda()
        text = text.cuda()
        _, h_fusion = fusion_model(paddle.concat((args.gamma * image, text), 1))
        codes = paddle.sign(h_fusion)
        test_codes.append(codes.numpy())
    test_codes = np.concatenate(test_codes)

    map = utils.calc_map(test_codes, database_codes, test_labels.numpy(), database_labels.numpy())
    print(f'mAP: {map}')
    
if __name__ == '__main__':
    args = parse_arguments()

    os.environ['FLAGS_cudnn_deterministic'] = 'True'
    paddle.device.set_device('gpu:1')

    paths = ''
    if args.dataset == 'flickr':
        paths = '/home1/ljy/dataset/mir_cnn_twt.mat'
    elif args.dataset == 'nuswide':
        paths = '/home1/ljy/dataset/nus_cnn_twt.mat'
    elif args.dataset == 'coco':
        paths = '/home1/ljy/dataset/coco_cnn_twt.mat'
    else:
        print("This dataset does not exist!")
    data = h5py.File(paths, 'r')

    I_tr = paddle.to_tensor(data=data['I_tr'][:].T, dtype='float32')
    T_tr = paddle.to_tensor(data=data['T_tr'][:].T, dtype='float32')
    L_tr = paddle.to_tensor(data=data['L_tr'][:].T, dtype='float32')
    I_db = paddle.to_tensor(data=data['I_db'][:].T, dtype='float32')
    T_db = paddle.to_tensor(data=data['T_db'][:].T, dtype='float32')
    L_db = paddle.to_tensor(data=data['L_db'][:].T, dtype='float32')
    I_te = paddle.to_tensor(data=data['I_te'][:].T, dtype='float32')
    T_te = paddle.to_tensor(data=data['T_te'][:].T, dtype='float32')
    L_te = paddle.to_tensor(data=data['L_te'][:].T, dtype='float32')
    
    BCELoss = paddle.nn.BCELoss()

    for t in range(args.times):

        train_loader, train_labels, label_dim, text_dim, num_train, mean_image, mean_text = generate_train_ds(I_tr, T_tr, L_tr)
        train_labels = train_labels.cuda()
        test_loader, test_labels = generate_test_database_ds(I_te, T_te, L_te)
        database_loader, database_labels = generate_test_database_ds(I_db, T_db, L_db)
        print('Data loader has been generated!Image dimension = 4096, text dimension = %d, label dimension = %d.' % (text_dim, label_dim))

        for args.bit in [32]:
            print('nc = %d' % args.nc)
            print('gamma = %f' % args.gamma)
            print('lamda = %f' % args.lamda)
            print('a = %f' % args.a)
            print('p1 = %f' % args.p1)
            print('p2 = %f' % args.p2)
            print('bit = %d' % args.bit)

            label_model = LabelModule(label_dim, args.bit)

            autoencoder_gcn_model = AutoEncoderGcnModule(4096, text_dim)

            fusion_model = FusionModule(4096 + text_dim, args.bit)

            lr_l = 0.1
            lr_a = 0.1
            lr_f = 0.1
            # lr_decay = np.exp(np.linspace(0, -8, args.Epoch_num))
            lr_decay = np.linspace(1, 0.01, args.Epoch_num)

            map_max = 0
            map_max_i = 0
            map_max_t = 0
            Losses = []
            path = './checkpoint/GCIMH_' + str(args.dataset) + '_' + str(args.bit) + '.pdparams'
            if args.train:
                print('Training...')
                for Epoch in range(args.Epoch_num):
                    print('Epoch: %d' % (Epoch + 1))

                    # Set the model to training mode
                    label_model.train()
                    autoencoder_gcn_model.train()
                    fusion_model.train()

                    # set the optimizer
                    label_optimizer = paddle.optimizer.Adam(parameters=label_model.parameters(), learning_rate=lr_l * lr_decay[Epoch])
                    autoencoder_gcn_optimizer = paddle.optimizer.Adam(parameters=autoencoder_gcn_model.parameters(), learning_rate=lr_a * lr_decay[Epoch])
                    fusion_optimizer = paddle.optimizer.Adam(parameters=fusion_model.parameters(), learning_rate=lr_f * lr_decay[Epoch])

                    # features of label modality
                    for epoch in range(5):
                        for i in range(20):
                            s = utils.calculate_s(train_labels, train_labels)
                            all_h_label = label_model(train_labels)
                            loss_label = negative_log_likelihood_similarity_loss1(all_h_label, all_h_label.detach(), s, args.bit) \
                                + args.beta * quantization_loss1(all_h_label)
                            label_optimizer.clear_grad()
                            loss_label.backward()
                            label_optimizer.step()
                        print('Loss label: %.4f' % loss_label.numpy())

                    # autoencoder
                    for epoch in range(5):
                        Loss1 = 0
                        Loss2 = 0
                        for i, (image, text, label, m1, m2) in enumerate(train_loader):
                            label = label.cuda()
                            image = image.cuda()
                            text = text.cuda()
                            m1 = m1.cuda()
                            m2 = m2.cuda()
                            # construct the graph for this batch
                            graph = paddle.matmul(label, label.t())
                            m = args.c * paddle.tile(paddle.transpose(paddle.multiply(m1, m2), [1, 0]), [250, 1]) + (1 - args.c) * (1 - paddle.tile(paddle.transpose(paddle.multiply(m1, m2), [1, 0]), [250, 1]))
                            graph = paddle.multiply(graph, m)
                            g = graph - paddle.multiply(graph, paddle.eye(label.shape[0]).cuda())
                            p = paddle.multiply(m1, m2) * args.p1 + (1 - paddle.multiply(m1, m2)) * args.p2
                            graph = (1 - p) * (g / paddle.sum(g + 1e-6, axis=1, keepdim=True) + paddle.diag((paddle.sum(g, axis=1) < 0.5).astype('float32'))) + p * paddle.eye(label.shape[0]).cuda()
                            input_image = deepcopy(image)
                            input_text = deepcopy(text)
                            output_image, output_text, _ = autoencoder_gcn_model(graph, paddle.concat((args.gamma * input_image, input_text), 1))
                            loss1 = args.lamda * paddle.mean(paddle.multiply(output_image - args.gamma * image, m1) ** 2)
                            loss2 = paddle.nn.functional.binary_cross_entropy(paddle.multiply(output_text, m2), paddle.multiply(text, m2))
                            loss = loss1 + loss2
                            Loss1 += loss1.numpy()
                            Loss2 += loss2.numpy()
                            autoencoder_gcn_optimizer.clear_grad()
                            loss.backward()
                            autoencoder_gcn_optimizer.step()
                        print('Loss antoencoder image: %.4f, Loss antoencoder text: %.4f' % (Loss1 / i, Loss2 / i))
                    
                    for epoch in range(5):
                        Loss1 = 0
                        Loss2 = 0
                        Loss3 = 0
                        for i, (image, text, label, m1, m2) in enumerate(train_loader):
                            label = label.cuda()
                            image = image.cuda()
                            text = text.cuda()
                            m1 = m1.cuda()
                            m2 = m2.cuda()
                            s = utils.calculate_s(label, train_labels)
                            # construct the graph for this batch
                            graph = paddle.matmul(label, label.t())
                            m = args.c * paddle.tile(paddle.transpose(paddle.multiply(m1, m2), [1, 0]), [250, 1]) + (1 - args.c) * (1 - paddle.tile(paddle.transpose(paddle.multiply(m1, m2), [1, 0]), [250, 1]))                            
                            graph = paddle.multiply(graph, m)
                            g = graph - paddle.multiply(graph, paddle.eye(label.shape[0]).cuda())
                            p = paddle.multiply(m1, m2) * args.p1 + (1 - paddle.multiply(m1, m2)) * args.p2
                            graph = (1 - p) * (g / paddle.sum(g + 1e-6, axis=1, keepdim=True) + paddle.diag((paddle.sum(g, axis=1) < 0.5).astype('float32'))) + p * paddle.eye(label.shape[0]).cuda()
                            _, _, latent = autoencoder_gcn_model(graph, paddle.concat((args.gamma * image, text), 1))
                            fusion, h_fusion = fusion_model(paddle.concat((args.gamma * image, text), 1))
                            loss1 = args.a * correspondence_loss(fusion, latent)
                            loss2 = args.alpha * negative_log_likelihood_similarity_loss1(h_fusion, all_h_label.detach(), s, args.bit)
                            loss3 = args.beta * quantization_loss1(h_fusion)
                            loss = loss1 + loss2 + loss3
                            Loss1 += loss1.numpy()
                            Loss2 += loss2.numpy()
                            Loss3 += loss3.numpy()
                            fusion_optimizer.clear_grad()
                            loss.backward()
                            fusion_optimizer.step()
                        print('Latent Loss: %.4f, Similarity Loss: %.4f, Quantization Loss: %.10f' % (Loss1 / i, Loss2 / i, Loss3 / i))
                    evaluate()
                paddle.save({'label_model': label_model.state_dict(), 'autoencoder_gcn_model': autoencoder_gcn_model.state_dict(), 'fusion_model': fusion_model.state_dict()}, path)
                print('end of training')
            else:
                print('Testing...')
                print("start to load model")
                checkpoint = paddle.load(path)
                label_model.load_state_dict(checkpoint['label_model'])
                autoencoder_gcn_model.load_state_dict(checkpoint['autoencoder_gcn_model'])
                fusion_model.load_state_dict(checkpoint['fusion_model'])
                print("model has been loaded")
                evaluate()
                print('end of testing')
