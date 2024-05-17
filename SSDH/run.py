import argparse
import os
from evaluate import logger
import ssdh
from data.data_loader import load_data
from model_loader import load_model
import paddle

multi_labels_dataset = [
    'flickr',
]

num_features = {
    'alexnet': 4096,
}

paddle.device.set_device('gpu:7')

def run():
    # Load configuration
    args = load_config()
    log = logger()
    args.checkpoint = './checkpoint/SSDH_' + str(args.dataset) + '_' + str(args.code_length) + '.pdparams'

    # Load dataset
    test_dataloader, train_dataloder, database_dataloader, test_label, train_label, database_label = load_data(args.dataset,
                                                                                                                args.data_path,
                                                                                                                args.batch_size,
                                                                                                                args.num_workers,
                                                                                                                )

    multi_labels = args.dataset in multi_labels_dataset
    if args.train:
        ssdh.train(
            log,
            train_dataloder,
            test_dataloader,
            database_dataloader,
            multi_labels,
            args.code_length,
            num_features[args.arch],
            args.alpha,
            args.beta,
            args.max_iter,
            args.arch,
            args.lr,
            args.verbose,
            args.evaluate_interval,
            args.snapshot_interval,
            args.topk,
            args.checkpoint,
            test_label,
            train_label,
            database_label
        )
    elif args.evaluate:
        model = load_model(args.arch, args.code_length)
        model.load_snapshot(args.checkpoint)
        model.eval()
        mAP = ssdh.evaluate(
            log,
            model,
            test_dataloader,
            database_dataloader,
            args.code_length,
            args.topk,
            multi_labels,
            test_label,
            train_label,
            database_label
            )
        log.info('[Inference map:{:.4f}]'.format(mAP.item()))
    else:
        raise ValueError('Error configuration, please check your config, using "train" or "evaluate".')


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='SSDH_PaddlePaddle')
    parser.add_argument('--dataset', default = 'flickr', help='Dataset name.')
    parser.add_argument('--data_path', default = '/home/zhangyh/code/ljy/dataset/mir25_crossmodal.h5', help='Path of dataset')
    parser.add_argument('-c', '--code-length', default=32, type=int,
                        help='Binary hash code length.(default: 12)')
    parser.add_argument('-T', '--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('-l', '--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-3)')
    parser.add_argument('-w', '--num-workers', default=4, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='Batch size.(default: 24)')
    parser.add_argument('-a', '--arch', default='alexnet', type=str,
                        help='CNN architecture.(default: vgg16)')
    parser.add_argument('-k', '--topk', default=5000, type=int,
                        help='Calculate map of top k.(default: 5000)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print log.')
    parser.add_argument('--train', action='store_true',
                        help='Training mode.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate mode.')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('-e', '--evaluate-interval', default=500, type=int,
                        help='Interval of evaluation.(default: 500)')
    parser.add_argument('-s', '--snapshot-interval', default=800, type=int,
                        help='Interval of evaluation.(default: 800)')
    parser.add_argument('-C', '--checkpoint', default=None, type=str,
                        help='Path of checkpoint.')
    parser.add_argument('--alpha', default=2, type=float,
                        help='Hyper-parameter.(default:2)')
    parser.add_argument('--beta', default=2, type=float,
                        help='Hyper-parameter.(default:2)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    run()
