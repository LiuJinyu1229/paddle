import os.path as osp

import paddle
import paddle.nn.functional as F

import datasets
import settings
from metric import compress_wiki, compress, calculate_map
from models import ImgNet, TxtNet

paddle.seed(1)
paddle.device.set_device('gpu:0')

class Session:
    def __init__(self):
        self.logger = settings.logger

        # paddle.set_device('gpu:%d' % settings.GPU_ID)

        if settings.DATASET == "WIKI":
            self.train_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True,
                                               transform=datasets.wiki_train_transform)
            self.test_dataset = datasets.WIKI(root=settings.DATA_DIR, train=False,
                                              transform=datasets.wiki_test_transform)
            self.database_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True,
                                                  transform=datasets.wiki_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = paddle.io.DataLoader(dataset=self.train_dataset,
                                                 batch_size=settings.BATCH_SIZE,
                                                 shuffle=True,
                                                 num_workers=settings.NUM_WORKERS,
                                                 drop_last=True)

        self.test_loader = paddle.io.DataLoader(dataset=self.test_dataset,
                                                batch_size=settings.BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=settings.NUM_WORKERS)

        self.database_loader = paddle.io.DataLoader(dataset=self.database_dataset,
                                                    batch_size=settings.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=settings.NUM_WORKERS)
        txt_feat_len = datasets.txt_feat_len
        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        if settings.DATASET == "WIKI":
            self.opt_I = paddle.optimizer.SGD(parameters=self.CodeNet_I.parameters(), learning_rate=settings.LR_IMG, weight_decay=settings.WEIGHT_DECAY)

        self.opt_T = paddle.optimizer.SGD(parameters=self.CodeNet_T.parameters(), learning_rate=settings.LR_TXT, weight_decay=settings.WEIGHT_DECAY)
        self.best = 0

    def train(self, epoch):
        self.FeatNet_I.train()

        self.CodeNet_I.train()
        self.CodeNet_T.train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, txt, labels, _) in enumerate(self.train_loader):
            batch_size = img.shape[0]
            # txt = txt.astype('float64')
            img = paddle.to_tensor(img)
            txt = paddle.to_tensor(txt)
            self.opt_I.clear_grad()
            self.opt_T.clear_grad()

            (_, F_I), _, _, _ = self.FeatNet_I(img)
            F_T = txt
            _, hid_I, code_I, decoded_t = self.CodeNet_I(img)
            _, hid_T, code_T, decoded_i = self.CodeNet_T(txt)
            F_I = paddle.nn.functional.normalize(F_I)
            S_I = F_I.matmul(F_I.t())
            S_I = S_I * 2 - 1
            F_T = paddle.nn.functional.normalize(F_T)
            S_T = F_T.matmul(F_T.t())
            S_T = S_T * 2 - 1

            B_I = paddle.nn.functional.normalize(code_I)
            B_T = paddle.nn.functional.normalize(code_T)

            BI_BI = B_I.matmul(B_I.t())
            BT_BT = B_T.matmul(B_T.t())
            BI_BT = B_I.matmul(B_T.t())

            S_tilde = settings.ALPHA * S_I + (1 - settings.ALPHA) * S_T
            S = settings.K * S_tilde

            loss1 = paddle.nn.functional.mse_loss(BT_BT, S)
            loss2 = paddle.nn.functional.mse_loss(BI_BT, S)
            loss3 = paddle.nn.functional.mse_loss(BI_BI, S)
            loss31 = paddle.nn.functional.mse_loss(BI_BI, settings.K * S_I)
            S_T = S_T.astype('float32')
            loss32 = paddle.nn.functional.mse_loss(BT_BT, settings.K * S_T)

            diagonal = paddle.diag(BI_BT)
            all_1 = paddle.full_like(paddle.rand((batch_size,)), 1)
            loss4 = paddle.nn.functional.mse_loss(diagonal, settings.K * all_1)
            loss5 = paddle.nn.functional.mse_loss(decoded_i, F_I)
            F_T = F_T.astype('float32')
            loss6 = paddle.nn.functional.mse_loss(decoded_t, F_T)
            loss7 = paddle.nn.functional.mse_loss(BI_BT, BI_BT.t())
            loss = 1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 1 * loss5 + 1 * loss6 + 2 * loss7 + settings.ETA * (
                        loss31 + loss32)

            # Wiki
            # 128bit 435 662;432 661;434 667; 433，663;
            # 64 bit 440 658;438 660;433 660;
            # 32 bit 430 650;422 658;420 665;
            # 16 bit 394 617;416 644;416 639;

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss1: %.4f Loss2: %.4f Loss3: %.4f '
                    'Loss4: %.4f '
                    'Loss5: %.4f Loss6: %.4f '
                    'Loss7: %.4f '
                    'Total Loss: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(), loss3.item(),
                        loss4.item(),
                        loss5.item(), loss6.item(),
                        loss7.item(),
                        loss.item()))

    def eval(self):

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval()
        self.CodeNet_T.eval()
        if settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.CodeNet_I, self.CodeNet_T,
                                                                   self.database_dataset, self.test_dataset)
            K = [1, 200, 400, 500, 1000, 1500, 2000]
        # if settings.EVAL:
        #     MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
        #     MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
        #     self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        #     self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        #     self.logger.info('--------------------------------------------------------------------')
        MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
        MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('--------------------------------------------------------------------')
        if MAP_I2T + MAP_T2I > self.best and not settings.EVAL:
            self.save_checkpoints()
            self.best = MAP_T2I + MAP_I2T
            self.logger.info("#########is best:%.3f #########" % self.best)

    def save_checkpoints(self):
        file_path = 'DSAH_%s_%d.pdparams' % (settings.DATASET, settings.CODE_LEN)
        ckp_path = osp.join(settings.MODEL_DIR, file_path)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
        }
        paddle.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self):
        file_path = 'DSAH_%s_%d.pdparams' % (settings.DATASET, settings.CODE_LEN)
        ckp_path = osp.join(settings.MODEL_DIR, file_path)
        try:
            obj = paddle.load(ckp_path)
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.set_state_dict(obj['ImgNet'])
        self.CodeNet_T.set_state_dict(obj['TxtNet'])

def main():
    sess = Session()

    if settings.EVAL == False:
        print('**********Start testing...**********')
        sess.load_checkpoints()
        sess.eval()
        print('**********Test finished.**********')

    else:
        print('**********Start training...**********')
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval()
            # save the model
        sess.eval()
        settings.EVAL = True
        print('**********Training finished.********')


if __name__ == '__main__':
    main()
