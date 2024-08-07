import logging
import os.path as osp
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Training mode')
args = parser.parse_args()

# EVAL = True: just test, EVAL = False: train and eval
EVAL = args.train

# dataset: 'WIKI'

DATASET = 'WIKI'

if DATASET == 'WIKI':
    DATA_DIR = '/home1/ljy/dataset/wikipedia/images'
    LABEL_DIR = '/home1/ljy/dataset/wikipedia/raw_features.mat'
    TRAIN_LABEL = '/home1/ljy/dataset/wikipedia/trainset_txt_img_cat.list'
    TEST_LABEL = '/home1/ljy/dataset/wikipedia/testset_txt_img_cat.list'
    # loss = 1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 1 * loss5 + 1 * loss6 + 2 * loss7 + 0.1 * loss31 + 0.1 * loss32
    NUM_EPOCH = 200
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 40

K = 1.5
ETA = 0.1
ALPHA = 0.9

BATCH_SIZE = 32
CODE_LEN = 16
MOMENTUM = 0.7
WEIGHT_DECAY = 5e-4

GPU_ID = 4
NUM_WORKERS = 8
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('CODE_LEN = %d' % CODE_LEN)
logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('ALPHA = %.4f' % ALPHA)
logger.info('K = %.4f' % K)
logger.info('ETA = %.4f' % ETA)


logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)


logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)

logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)

logger.info('--------------------------------------------------------------------')
