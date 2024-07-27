import numpy as np
import paddle
import argparse
import time
import sys
import scipy.io as sio
from model_transformer import GMMH, L2H_Prototype
from paddle.io import DataLoader
import utils
from data import *

paddle.device.set_device('gpu:5')

def train(args, dset):
    print('=' * 30)
    print('Training Stage...')
    print('Train size: %d' % dset.I_tr.shape[0])
    assert dset.I_tr.shape[0] == dset.T_tr.shape[0]
    assert dset.I_tr.shape[0] == dset.L_tr.shape[0]
    loss_l2 = paddle.nn.MSELoss()
    # loss_cl = torch.nn.MultiLabelSoftMarginLoss()
    l2h = L2H_Prototype(args=args)
    l2h.train()
    gmmh = GMMH(args=args)
    gmmh.train()
    optimizer_L2H = paddle.optimizer.Adam(learning_rate=args.lr, parameters=l2h.parameters())
    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=gmmh.parameters())
    start_time = time.time() * 1000
    _, COO_matrix = utils.get_COO_matrix(dset.L_tr)
    COO_matrix = paddle.to_tensor(data=COO_matrix, dtype='float32')
    train_label = paddle.to_tensor(data=dset.L_tr, dtype='float32')
    for epoch in range(args.epochs_pre):
        prototype, code, pred = l2h(train_label)
        optimizer_L2H.clear_grad()
        B = paddle.sign(x=code)
        prototype_norm = paddle.nn.functional.normalize(x=prototype)
        prototype_norm = prototype_norm.astype('float32')
        recon_loss = loss_l2(paddle.nn.functional.sigmoid(pred), train_label) * args.param_recon_pre
        sign_loss = loss_l2(code, B) * args.param_sign_pre
        bal_loss = paddle.sum(x=code) / code.shape[0] * args.param_bal_pre
        static_loss = loss_l2(prototype_norm.mm(mat2=prototype_norm.t()), COO_matrix) * args.param_static_pre
        loss = recon_loss + sign_loss + bal_loss + static_loss
        loss.backward()
        optimizer_L2H.step()
    l2h.eval()
    B_tr = np.sign(l2h(train_label)[1].cpu().numpy())
    map_train = utils.calculate_map(B_tr, B_tr, dset.L_tr, dset.L_tr)
    print('Training MAP: %.4f' % map_train)
    print('=' * 30)
    train_loader = DataLoader(my_dataset(dset.I_tr, dset.T_tr, dset.L_tr, B_tr=B_tr), 
                              batch_size=args.batch_size, 
                              shuffle=True)
    for epoch in range(args.epochs):
        for i, (idx, img_feat, txt_feat, label, B_gnd) in enumerate(
            train_loader):
            img_feat = img_feat.cuda()
            txt_feat = txt_feat.cuda()
            label = label.astype('float32').cuda()
            B_gnd = B_gnd.cuda()
            aff_label = utils.affinity_tag_multi(label, label)
            aff_label = paddle.to_tensor(aff_label).cuda()
            optimizer.clear_grad()
            H, pred = gmmh(img_feat, txt_feat)
            H_norm = paddle.nn.functional.normalize(x=H)
            clf_loss = loss_l2(paddle.nn.functional.sigmoid(pred), label)
            sign_loss = loss_l2(H, B_gnd)
            similarity_loss = loss_l2(H_norm.mm(mat2=H_norm.t()), aff_label)
            loss = clf_loss * args.param_clf + sign_loss * args.param_sign + similarity_loss * args.param_sim
            loss.backward()
            optimizer.step()
            if i + 1 == len(train_loader) and (epoch + 1) % 2 == 0:
                print('Epoch [%3d/%3d], Loss: %.4f, Loss-C: %.4f, Loss-B: %.4f, Loss-S: %.4f'
                    % (epoch + 1, args.epochs, loss.item(), 
                       clf_loss.item() * args.param_clf, 
                       sign_loss.item() * args.param_sign,
                       similarity_loss.item() * args.param_sim))
    end_time = time.time() * 1000
    elapsed = (end_time - start_time) / 1000
    print('Training Time: ', elapsed)
    gmmh_path = 'checkpoint/BSTH_' + args.dataset + '_' + str(args.nbit) + '.paparams'
    paddle.save(gmmh.state_dict(), gmmh_path)
    return gmmh


def eval(model, dset, args):
    model.eval()
    print('=' * 30)
    print('Testing Stage...')
    print('Retrieval size: %d' % dset.I_db.shape[0])
    assert dset.I_db.shape[0] == dset.T_db.shape[0]
    assert dset.I_db.shape[0] == dset.L_db.shape[0]
    retrieval_loader = DataLoader(my_dataset(dset.I_db, dset.T_db, dset.L_db), 
                                  batch_size=args.eval_batch_size, 
                                  shuffle=False,
                                  num_workers=0)
    retrievalP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, _) in enumerate(retrieval_loader):
        img_feat = img_feat
        txt_feat = txt_feat
        H, _ = model(img_feat, txt_feat)
        retrievalP.append(H.cpu().numpy())
    retrievalH = np.concatenate(retrievalP)
    retrievalCode = np.sign(retrievalH)
    end_time = time.time() * 1000
    retrieval_time = end_time - start_time
    print('Query size: %d' % dset.I_te.shape[0])
    assert dset.I_te.shape[0] == dset.T_te.shape[0]
    assert dset.I_te.shape[0] == dset.L_te.shape[0]
    val_loader = DataLoader(my_dataset(dset.I_te, dset.T_te, dset.L_te), 
                            batch_size=args.eval_batch_size, 
                            shuffle=False, 
                            num_workers=0)
    valP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, _) in enumerate(val_loader):
        img_feat = img_feat
        txt_feat = txt_feat
        H, _ = model(img_feat, txt_feat)
        valP.append(H.cpu().numpy())
    valH = np.concatenate(valP)
    valCode = np.sign(valH)
    end_time = time.time() * 1000
    query_time = end_time - start_time
    print('[Retrieval time] %.4f, [Query time] %.4f' % (retrieval_time / 1000, query_time / 1000))
    map_eval = utils.calculate_map(valCode, retrievalCode, dset.L_te, dset.L_db)
    print('Evaluation MAP: %.4f' % map_eval)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GMMH', help='Use GMMH.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of student epochs to train.')
    parser.add_argument('--epochs_pre', type=int, default=100, help='Epoch to learn the hashcode.')
    parser.add_argument('--nbit', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=1, help='"nhead" in Transformer.')
    parser.add_argument('--num_layer', type=int, default=2, help='"num_layer" in Transformer.')
    parser.add_argument('--trans_act', type=str, default='gelu', help='"activation" in Transformer.')
    parser.add_argument('--dataset', type=str, default='flickr', help='coco/nuswide/flickr')
    parser.add_argument('--classes', type=int, default=24)
    parser.add_argument('--image_dim', type=int, default=4096)
    parser.add_argument('--text_dim', type=int, default=1386)
    parser.add_argument('--img_hidden_dim', type=list, default=[2048, 128], help='Construct imageMLP')
    parser.add_argument('--txt_hidden_dim', type=list, default=[1024, 128], help='Construct textMLP')
    parser.add_argument('--L2H_hidden_dim', type=list, default=[1024, 1024], help='Construct L2H')
    parser.add_argument('--param_recon_pre', type=float, default=0.001)
    parser.add_argument('--param_sign_pre', type=float, default=100)
    parser.add_argument('--param_bal_pre', type=float, default=0.01)
    parser.add_argument('--param_static_pre', type=float, default=1)
    parser.add_argument('--param_clf', type=float, default=1, help='')
    parser.add_argument('--param_sign', type=float, default=0.01, help='nuswide: 0.0001/')
    parser.add_argument('--param_sim', type=float, default=1)
    parser.add_argument('--save_flag', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--train', action='store_true', help='Training mode')
    args = parser.parse_args()
    utils.seed_setting(args.seed)
    dset = load_data(args.dataset)
    print('Train size: %d, Retrieval size: %d, Query size: %d' % (dset.I_tr.shape[0], dset.I_db.shape[0], dset.I_te.shape[0]))
    print('Image dimension: %d, Text dimension: %d, Label dimension: %d' % (dset.I_tr.shape[1], dset.T_tr.shape[1], dset.L_tr.shape[1]))
    args.image_dim = dset.I_tr.shape[1]
    args.text_dim = dset.T_tr.shape[1]
    args.classes = dset.L_tr.shape[1]
    args.img_hidden_dim.insert(0, args.image_dim)
    args.txt_hidden_dim.insert(0, args.text_dim)
    args.L2H_hidden_dim.insert(0, args.classes)
    args.L2H_hidden_dim.append(args.nbit)
    if args.train:
        model = train(args, dset)
    else:
        print('Load model...')
        model = GMMH(args=args)
        model_path = 'checkpoint/BSTH_' + args.dataset + '_' + str(args.nbit) + '.paparams'
        model.set_state_dict(paddle.load(model_path))
        print('Model loaded.')
    eval(model, dset, args)
