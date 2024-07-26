import datetime
import time
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader

import os
import argparse
import logging
import time
import numpy as np
from tqdm import tqdm

import utils.data_processing as dp
import utils.adsh_loss as al
import utils.cnn_model as cnn_model
import utils.subset_sampler as subsetsampler
import utils.calc_hr as calc_hr

paddle.device.set_device('gpu:1')

parser = argparse.ArgumentParser(description="ADSH")
parser.add_argument('--bit', default='32', type=str,
                    help='binary code length (default: 12,24,32,48)')
parser.add_argument('--gpu', default='1', type=str,
                    help='selected gpu (default: 1)')
parser.add_argument('--arch', default='alexnet', type=str,
                    help='model name (default: alexnet)')
parser.add_argument('--max-iter', default=50, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=3, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='batch size (default: 64)')
parser.add_argument('--data_set', default='flickr', type=str, help='data set')
parser.add_argument('--num-samples', default=2000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--gamma', default=200, type=int,
                    help='hyper-parameter: gamma (default: 200)')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--train', action='store_true', 
                    help='whether to train the model (default: True)')

def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def calc_sim(database_label, train_label):
    S = (paddle.matmul(database_label, paddle.transpose(train_label, [1, 0])) > 0).astype('float32')
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S

def calc_loss(V, U, S, code_length, select_index, gamma):
    V = paddle.to_tensor(V, dtype='float32')
    U = paddle.to_tensor(U, dtype='float32')
    S = paddle.to_tensor(S, dtype='float32')
    num_database = V.shape[0]
    square_loss = (paddle.matmul(U, paddle.transpose(V, [1, 0])) - code_length*S) ** 2
    V_omega = V[select_index, :]
    quantization_loss = (U-V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)
    return loss

def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = paddle.to_tensor(data_input)
        output = model(data_input)
        B[data_ind.numpy(), :] = paddle.sign(output).numpy()
    return B

def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 20, 30, 40, 50]
    if iter in update_list:
        optimizer.set_lr(optimizer.get_lr() / 10)

def adsh_algo():
    paddle.seed(0)

    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_samples = opt.num_samples
    gamma = opt.gamma
    code_length = int(opt.bit)
    topk = 5000
    checkpoint = './checkpoint/ADSH_' + opt.data_set + '_' + str(opt.bit) + '.pdparams'

    logger.info(opt)
    logger.info(code_length)

    '''
    dataset preprocessing
    '''
    dset_database = dp.DatasetProcessing(train=False, database=True, transform=dp.test_transform)
    dset_test = dp.DatasetProcessing(train=False, database=False, transform=dp.test_transform)
    num_database = len(dset_database)
    num_test = len(dset_test)
    database_labels = dset_database.label
    test_labels = dset_test.label

    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_length)
    adsh_loss = al.ADSHLoss(gamma, code_length, num_database)
    optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)

    V = np.zeros((num_database, code_length))

    if opt.train:
        print("start training")
        model.train()
        for iter in range(max_iter):
            iter_time = time.time()
            '''
            sampling and construct similarity matrix
            '''
            select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
            _sampler = subsetsampler.SubsetSampler(dset_database, select_index)
            trainloader = paddle.io.DataLoader(_sampler, batch_size=batch_size, shuffle=False, num_workers=4)
            '''
            learning deep neural network: feature learning
            '''
            # sample_label = database_labels.index_select(0, paddle.to_tensor(np.array(select_index)))
            sample_label = np.array([database_labels[i] for i in select_index])
            database_labels = paddle.to_tensor(database_labels[:], dtype='float32')
            sample_label = paddle.to_tensor(sample_label, dtype='float32')
            Sim = calc_sim(sample_label, database_labels)
            U = np.zeros((num_samples, code_length), dtype='float32')
            for epoch in range(epochs):
                trainloader = tqdm(trainloader)
                for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                    batch_size_ = train_label.shape[0]
                    u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration+1)*batch_size)) - 1, batch_size_, dtype=int)
                    train_input = paddle.to_tensor(train_input)

                    output = model(train_input)
                    S = paddle.to_tensor(np.array([Sim[i] for i in u_ind]), dtype='float32')
                    # S = Sim.index_select(0, paddle.to_tensor(u_ind))
                    U[u_ind, :] = output.cpu().numpy()

                    optimizer.clear_grad()
                    loss = adsh_loss(output, V, S, V[batch_ind.numpy(), :])
                    loss.backward()
                    optimizer.step()
            adjusting_learning_rate(optimizer, iter)

            '''
            learning binary codes: discrete coding
            '''
            barU = np.zeros((num_database, code_length))
            barU[select_index, :] = U
            Q = -2*code_length*Sim.numpy().transpose().dot(U) - 2 * gamma * barU
            for k in range(code_length):
                sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
                V_ = V[:, sel_ind]
                Uk = U[:, k]
                U_ = U[:, sel_ind]
                V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
            iter_time = time.time() - iter_time
            loss_ = calc_loss(V, U, Sim.numpy(), code_length, select_index, gamma)
            logger.info('[Iteration: %3d/%3d][Train Loss: %.4f]', iter, max_iter, loss_)
        paddle.save(model.state_dict(), checkpoint)
        print('---finish training---')
    else:
        print('---loading trained model---')
        model.set_state_dict(paddle.load(checkpoint))
        print('---finish loading---')
        databaseloader = DataLoader(dset_database, batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
        V = encode(model, databaseloader, num_database, code_length)
    print('---start evaluation---')
    model.eval()
    testloader = paddle.io.DataLoader(dset_test, batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
    qB = encode(model, testloader, num_test, code_length)
    rB = V
    map = calc_hr.calc_map(qB, rB, paddle.to_tensor(np.array(test_labels)).numpy(), paddle.to_tensor(np.array(database_labels)).numpy())
    topkmap = calc_hr.calc_topMap(qB, rB, paddle.to_tensor(np.array(test_labels)).numpy(), paddle.to_tensor(np.array(database_labels)).numpy(), topk)
    logger.info('[Evaluation: mAP: %.4f, top-%d mAP: %.4f]', map, topk, topkmap)
    print('---finish evaluation---')

if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    logdir = '-'.join(['log/', datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    adsh_algo()
