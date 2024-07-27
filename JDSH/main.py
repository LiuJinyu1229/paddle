from JDSH import JDSH
from utils import logger
from args import config
import paddle

paddle.device.set_device('gpu:5')

def log_info(logger, config):

    logger.info('--- Configs List---')
    logger.info('--- Dadaset:{}'.format(config.DATASET))
    logger.info('--- Train:{}'.format(config.TRAIN))
    logger.info('--- Bit:{}'.format(config.HASH_BIT))
    logger.info('--- Alpha:{}'.format(config.alpha))
    logger.info('--- Beta:{}'.format(config.beta))
    logger.info('--- Lambda:{}'.format(config.lamb))
    logger.info('--- Mu:{}'.format(config.mu))
    logger.info('--- Batch:{}'.format(config.BATCH_SIZE))
    logger.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    logger.info('--- Lr_TXT:{}'.format(config.LR_TXT))


def main():

    # log
    log = logger()
    log_info(log, config)

    Model = JDSH(log, config)

    if config.TRAIN == False:
        print("Testing...")
        Model.load_checkpoints(config.CHECKPOINT)
        Model.eval()
        print("Testing Done!")

    else:
        print("Training...")
        for epoch in range(config.NUM_EPOCH):
            Model.train(epoch)
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                Model.eval()
            # save the model
            if epoch + 1 == config.NUM_EPOCH:
                Model.save_checkpoints(config.CHECKPOINT)
        print("Training Done!")



if __name__ == '__main__':
    main()
