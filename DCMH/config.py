import warnings


class DefaultConfig(object):
    # data parameters
    dataset = 'flickr'
    data_path = '/home/zhangyh/code/ljy/dataset/FLICKR.mat'
    pretrain_model_path = '/home/zhangyh/code/ljy/dataset/imagenet-vgg-f.mat'
    training_size = 10000
    query_size = 2000
    database_size = 18015
    batch_size = 128

    # hyper-parameters
    max_epoch = 500
    gamma = 1
    eta = 1
    bit = 64  # final binary code length
    lr = 10 ** (-1.5)  # initial learning rate

    use_gpu = True

    valid = True

    print_freq = 2  # print info every N epoch

    result_dir = 'result'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
