import logging
import os.path as osp
import time
import argparse

# EVAL = True: just test, EVAL = False: train and eval
# EVAL = False
# EVAL = True

BATCH_SIZE = 32
CODE_LEN = 128
MOMENTUM = 0.7
WEIGHT_DECAY = 5e-4

GPU_ID = 1
NUM_WORKERS = 8
EPOCH_INTERVAL = 2

parser = argparse.ArgumentParser(description='Ours')
parser.add_argument('--dataname', type=str, default='flickr', help='Dataset name: flickr/coco/nuswide')
parser.add_argument('--Bit', default=32, help='hash bit', type=int)
parser.add_argument('--train', action='store_true', help='Training mode')
parser.add_argument('--GID', default=5, help='gpu id', type=int)
parser.add_argument('--Alpha', default=1, help='0 MIR, 1 NUS', type=float)
parser.add_argument('--Beta', default=1, help='0 MIR, 1 NUS', type=float)
args = parser.parse_args()
CODE_LEN = args.Bit
GPU_ID = args.GID
ALPHA = args.Alpha
BETA = args.Beta
EVAL = args.train

if args.dataname == 'flickr':
    DIR = '/home1/ljy/dataset/mir_cnn_twt.mat'
elif args.dataname == 'nuswide':
    DIR = '/home1/ljy/dataset/nus_cnn_twt.mat'
elif args.dataname == 'coco':
    DIR = '/home1/ljy/dataset/coco_cnn_twt.mat'
else:
    print('Dataname Error!')
    DIR = ''

DATASET = args.dataname
NUM_EPOCH = 200
LR_IMG = 0.005
LR_TXT = 0.005
EVAL_INTERVAL = 40

MODEL_DIR = './checkpoint'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("SCH_%Y%m%d%H%M%S", time.localtime(time.time()))
extension = '_%s_%d_log.txt' % (DATASET, CODE_LEN)
log_name = now + extension
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
logger.info('BETA = %.4f' % BETA)


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
