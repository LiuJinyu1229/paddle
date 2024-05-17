import argparse
from easydict import EasyDict as edict
import json

parser = argparse.ArgumentParser(description='JDSH')
parser.add_argument('--train', action='store_true', help='Training mode.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate mode.')
parser.add_argument('--Config', default='./config/JDSH_MIRFlickr.json', help='Configure path', type=str)
parser.add_argument('--Dataset', default='flickr', help='flickr or nuswide', type=str)
parser.add_argument('--Checkpoint', default='JDSH_flikcr_128.pdparams', help='checkpoint name', type=str)
parser.add_argument('--Bit', default=32, help='hash bit', type=int)

args = parser.parse_args()

# load basic settings
with open(args.Config, 'r') as f:
    config = edict(json.load(f))

# update settings
config.TRAIN = args.train
config.EVALUATE = args.evaluate
config.DATASET = args.Dataset
config.CHECKPOINT = 'JDSH_' + args.Dataset + '_' + str(args.Bit) + '.pdparams'
config.HASH_BIT = args.Bit
