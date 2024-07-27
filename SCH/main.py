import os.path as osp
import paddle
import paddle.nn.functional as F

import datasets
import settings
from utils import compress, calculate_top_map, calculate_map
from models import ImgNet, TxtNet

paddle.seed(1)

class Session:
    def __init__(self):
        self.logger = settings.logger

        paddle.set_device('gpu:%d' % settings.GPU_ID)

        self.train_dataset = datasets.MY_DATASET(train=True, transform=datasets.train_transform)
        self.test_dataset = datasets.MY_DATASET(train=False, database=False, transform=datasets.test_transform)
        self.database_dataset = datasets.MY_DATASET(train=False, database=True, transform=datasets.test_transform)

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
        self.maxfunc = paddle.nn.ReLU()

        self.opt_I = paddle.optimizer.Momentum(parameters=self.CodeNet_I.parameters(), learning_rate=settings.LR_IMG, momentum=settings.MOMENTUM,
                                           weight_decay=settings.WEIGHT_DECAY)

        self.opt_T = paddle.optimizer.Momentum(parameters=self.CodeNet_T.parameters(), learning_rate=settings.LR_TXT, momentum=settings.MOMENTUM,
                                           weight_decay=settings.WEIGHT_DECAY)
        self.best = 0
        
    def set_LR(self):
        self.opt_I = paddle.optimizer.Momentum(parameters=self.CodeNet_I.parameters(), learning_rate=settings.LR_IMG * 0.2, momentum=settings.MOMENTUM,
                                           weight_decay=settings.WEIGHT_DECAY)

        self.opt_T = paddle.optimizer.Momentum(parameters=self.CodeNet_T.parameters(), learning_rate=settings.LR_TXT * 0.2, momentum=settings.MOMENTUM,
                                           weight_decay=settings.WEIGHT_DECAY)
        
    def train(self, epoch):

        self.CodeNet_I.train()
        self.CodeNet_T.train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.4f, alpha for TxtNet: %.4f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, txt, labels, _) in enumerate(self.train_loader):
            img = paddle.to_tensor(img)
            txt = paddle.to_tensor(txt.astype('float32'))
            labels = paddle.to_tensor(labels.astype('float32'))
            self.opt_I.clear_grad()
            self.opt_T.clear_grad()

            _, _, B_I = self.CodeNet_I(img)
            _, _, B_T = self.CodeNet_T(txt)

            L = F.normalize(labels).matmul(F.normalize(labels).t())
            
            k = paddle.to_tensor(settings.CODE_LEN, dtype='float32')
            thresh = (1 - L) * k / 2
            width = 3
            up_thresh =  thresh
            low_thresh =  thresh - width
            low_thresh[low_thresh <= 0] = 0
            low_thresh[L == 0] = settings.CODE_LEN / 2
            
            low_flag = paddle.ones([settings.BATCH_SIZE, settings.BATCH_SIZE])
            up_flag = paddle.ones([settings.BATCH_SIZE, settings.BATCH_SIZE])
            low_flag[L == 1] = 0
            low_flag[L == 0] = settings.BETA
            up_flag[L == 0] = 0
            up_flag[L == 1] = settings.ALPHA

            BI_BI = (settings.CODE_LEN - B_I.matmul(B_I.t())) / 2
            BT_BT = (settings.CODE_LEN - B_T.matmul(B_T.t())) / 2
            BI_BT = (settings.CODE_LEN - B_I.matmul(B_T.t())) / 2
            BT_BI = (settings.CODE_LEN - B_T.matmul(B_I.t())) / 2

            # lower bound
            loss1 = (paddle.norm(F.relu(low_thresh - BI_BI) * low_flag) \
                    + paddle.norm(F.relu(low_thresh - BT_BT) * low_flag) \
                    + paddle.norm(F.relu(low_thresh - BT_BI) * low_flag) \
                    + paddle.norm(F.relu(low_thresh - BI_BT) * low_flag)) / (settings.BATCH_SIZE * settings.BATCH_SIZE)
            
            # upper bound
            loss2 = (paddle.norm(F.relu(BI_BI - up_thresh) * up_flag) \
                    + paddle.norm(F.relu(BT_BT - up_thresh) * up_flag) \
                    + paddle.norm(F.relu(BT_BI - up_thresh) * up_flag) \
                    + paddle.norm(F.relu(BI_BT - up_thresh) * up_flag)) / (settings.BATCH_SIZE * settings.BATCH_SIZE)
            
            loss = loss1 + loss2
        
            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss1: %.4f Loss2: %.4f'
                    'Total Loss: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(),
                        loss.item()))

    def eval(self, step=0, last=False):

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval()
        self.CodeNet_T.eval()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T, self.database_dataset, self.test_dataset)
            
        if settings.EVAL:
            MAP_I2T1 = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2I1 = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP of Image to Text: %.4f, MAP of Text to Image: %.4f' % (MAP_I2T1, MAP_T2I1))
            self.logger.info('--------------------------------------------------------------------')
            
            MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
            MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP of Image to Text: %.4f, MAP of Text to Image: %.4f' % (MAP_I2T, MAP_T2I))
            self.logger.info('--------------------------------------------------------------------')
            
        else:
            MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP of Image to Text: %.4f, MAP of Text to Image: %.4f' % (MAP_I2T, MAP_T2I))
            self.logger.info('--------------------------------------------------------------------')
            
        
        if MAP_I2T + MAP_T2I > self.best and not settings.EVAL:
            self.save_checkpoints(step=step, best=True)
            self.best = MAP_T2I + MAP_I2T
            self.logger.info("#########is best:%.4f #########" % self.best)

    def save_checkpoints(self, step, file_name='SCH_%s_%d_latest.pdparams' % (settings.DATASET, settings.CODE_LEN),
                         best=False):
        if best:
            file_name = 'SCH_%s_%d_best.pdparams' % (settings.DATASET, settings.CODE_LEN)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        paddle.save({
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='SCH_%s_%d_best.pdparams' % (settings.DATASET, settings.CODE_LEN)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = paddle.load(ckp_path)
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.set_state_dict(obj['ImgNet'])
        self.CodeNet_T.set_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def main():    
    sess = Session()

    if settings.EVAL == False:
        print("Testing...")
        sess.load_checkpoints()
        sess.eval()
        print("Testing Done!")

    else: 
        # settings.EVAL = True
        print("Training...")
        for epoch in range(settings.NUM_EPOCH):
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval(step=epoch + 1)
            # save the model
        settings.EVAL = True
        sess.load_checkpoints()
        sess.eval()
        print("Training Done!")


if __name__ == '__main__':
    main()
