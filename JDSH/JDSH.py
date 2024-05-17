import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import datasets
import os.path as osp
from models import ImgNet, TxtNet
from utils import compress, calculate_top_map, logger

class JDSH:
    def __init__(self, log, config):
        self.logger = log
        self.config = config
        paddle.seed(1)

        self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
        self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
        self.database_dataset = datasets.MIRFlickr(train=False, database=True,
                                                    transform=datasets.mir_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = paddle.io.DataLoader(dataset=self.train_dataset,
                                                batch_size=self.config.BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=self.config.NUM_WORKERS,
                                                drop_last=True)

        self.test_loader = paddle.io.DataLoader(dataset=self.test_dataset,
                                                batch_size=self.config.BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=self.config.NUM_WORKERS)

        self.database_loader = paddle.io.DataLoader(dataset=self.database_dataset,
                                                    batch_size=self.config.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=self.config.NUM_WORKERS)

        self.ImgNet = ImgNet(code_len=self.config.HASH_BIT)

        txt_feat_len = self.train_dataset.txt.shape[1]

        self.TxtNet = TxtNet(code_len=self.config.HASH_BIT, txt_feat_len=txt_feat_len)

        self.opt_I = paddle.optimizer.Momentum(learning_rate=self.config.LR_IMG, momentum=self.config.MOMENTUM,
                                            parameters=self.ImgNet.parameters(), weight_decay=self.config.WEIGHT_DECAY)
        self.opt_T = paddle.optimizer.Momentum(learning_rate=self.config.LR_TXT, momentum=self.config.MOMENTUM,
                                            parameters=self.TxtNet.parameters(), weight_decay=self.config.WEIGHT_DECAY)

        self.best_it = 0
        self.best_ti = 0

    def train(self, epoch):

        self.ImgNet.train()
        self.TxtNet.train()

        self.ImgNet.set_alpha(epoch)
        self.TxtNet.set_alpha(epoch)

        for idx, (img, txt, _, _) in enumerate(self.train_loader):

            img = paddle.to_tensor(img).cuda()
            txt = paddle.to_tensor(txt).cuda()

            self.opt_I.clear_grad()
            self.opt_T.clear_grad()

            F_I, hid_I, code_I = self.ImgNet(img)
            txt = paddle.to_tensor(txt, dtype='float32')
            code_T = self.TxtNet(txt)

            S = self.cal_similarity_matrix(F_I, txt)

            loss = self.cal_loss(code_I, code_T, S)

            loss.backward()

            self.opt_I.step()
            self.opt_T.step()

            if (idx + 1) % (len(self.train_dataset) // self.config.BATCH_SIZE / self.config.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                                 % (epoch + 1, self.config.NUM_EPOCH, idx + 1, len(self.train_dataset) // self.config.BATCH_SIZE,
                                     loss.item()))


    def eval(self):

        self.logger.info('--------------------Evaluation: mAP@50-------------------')

        self.ImgNet.eval()
        self.TxtNet.eval()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.ImgNet,
                                                          self.TxtNet, self.database_dataset, self.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        if (self.best_it + self.best_ti) < (MAP_I2T + MAP_T2I):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I

        self.logger.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (self.best_it, self.best_ti))
        self.logger.info('--------------------------------------------------------------------')

    def cal_similarity_matrix(self, F_I, txt):
        F_I = F.normalize(F_I)
        S_I = paddle.matmul(F_I, F_I.t())
        S_I = S_I * 2 - 1

        F_T = F.normalize(txt)
        S_T = paddle.matmul(F_T, F_T.t())
        S_T = S_T * 2 - 1

        S_high = F.normalize(S_I).mm(F.normalize(S_T).t())
        S_ = self.config.alpha * S_I + self.config.beta * S_T + self.config.lamb * (S_high + S_high.t()) / 2

        left = self.config.LOC_LEFT - self.config.ALPHA * self.config.SCALE_LEFT
        right = self.config.LOC_RIGHT + self.config.BETA * self.config.SCALE_RIGHT

        mask1 = S_ < left
        mask2 = S_ > right

        transformed_matrix = paddle.where(mask1, 1 + self.config.L1 * paddle.exp(self.config.MIN - S_), S_)
        transformed_matrix = paddle.where(mask2, 1 + self.config.L2 * paddle.exp(S_ - self.config.MAX), transformed_matrix)

        # S_[S_ < left] = (1 + self.config.L1 * paddle.exp(-(S_[S_ < left] - self.config.MIN))) \
        #                       * S_[S_ < left]
        # S_[S_ > right] = (1 + self.config.L2 * paddle.exp(S_[S_ > right] - self.config.MAX)) \
        #                        * S_[S_ > right]

        S = transformed_matrix  * self.config.mu

        return S
    
    def cal_loss(self, code_I, code_T, S):

        B_I = F.normalize(code_I, axis=1)
        B_T = F.normalize(code_T, axis=1)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        BT_BI = B_T.mm(B_I.t())

        loss1 = F.mse_loss(BI_BI, S)
        # loss2 = F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S) -(B_I * B_T).sum(dim=1).mean()
        loss2 = F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S) - paddle.sum(B_I * B_T, axis=1).mean()
        loss3 = F.mse_loss(BT_BT, S)

        loss = self.config.INTRA * loss1 + loss2 + self.config.INTRA * loss3

        return loss

    def save_checkpoints(self, file_name='latest.pdparams'):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.ImgNet.state_dict(),
            'TxtNet': self.TxtNet.state_dict(),
        }
        paddle.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')


    def load_checkpoints(self, file_name='latest.pdparams'):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        try:
            obj = paddle.load(ckp_path)
            self.ImgNet.set_state_dict(obj['ImgNet'])
            self.TxtNet.set_state_dict(obj['TxtNet'])
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError







