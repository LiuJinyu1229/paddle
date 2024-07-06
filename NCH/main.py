import argparse
import time
import utils
import paddle
from GCIC_ours import *

os.environ['FLAGS_cudnn_deterministic'] = 'True'
paddle.device.set_device('gpu:7')

def _main(config, logger, running_cnt):
    model = GCIC(config=config, logger=logger, running_cnt=running_cnt)
    if config.test == 0:
        logger.info('===========================================================================')
        logger.info('Training stage!')
        start_time = time.time() * 1000
        model.warmup()
        model.train()
        train_time = time.time() * 1000 - start_time
        logger.info('Training time: %.6f' % (train_time / 1000))
        logger.info('===========================================================================')
    logger.info('===========================================================================')
    logger.info('Testing stage!')
    start_time = time.time() * 1000
    model.test()
    test_time = time.time() * 1000 - start_time
    logger.info('Testing time: %.6f' % (test_time / 1000))
    logger.info('===========================================================================')


if __name__ == '__main__':
    utils.seed_setting(seed=2021)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr', help='Dataset: flickr/nuswide/coco')
    parser.add_argument('--alpha_train', type=float, default=0, help='Missing ratio of train set.')
    parser.add_argument('--alpha_query', type=float, default=0, help='Missing ratio of query set.')
    parser.add_argument('--beta_train', type=float, default=0.5)
    parser.add_argument('--beta_query', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=140) # flickr: 140, nuswide: 50, coco: 50
    parser.add_argument('--warmup_epochs', type=int, default=50) # 50
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--image_hidden_dim', type=int, default=2048)
    parser.add_argument('--text_hidden_dim', type=int, default=1024)
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--nbit', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--anchor_nums', type=int, default=300)
    parser.add_argument('--param_neighbour', type=float, default=0.1, help='Neighbour loss.')
    parser.add_argument('--param_sim', type=float, default=1, help='Similarity loss.')
    parser.add_argument('--param_sign', type=float, default=0.1, help='Sign loss.')
    parser.add_argument('--ANCHOR', type=str, default='random', help='Anchor choose!(random/balance)')
    parser.add_argument('--run_times', type=int, default=1)
    parser.add_argument('--test', type=int, default=0)
    logger = utils.logger()
    logger.info('===========================================================================')
    logger.info('Current File: {}'.format(__file__))
    config = parser.parse_args()
    utils.log_params(logger, vars(config))
    logger.info('===========================================================================')
    for i in range(config.run_times):
        _main(config=config, logger=logger, running_cnt=i + 1)
