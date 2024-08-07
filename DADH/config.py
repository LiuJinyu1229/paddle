import warnings

class Default(object):
    load_model_path = None  # load model path

    pretrain_model_path = './data/imagenet-vgg-f.mat'

    # visualization
    vis_env = 'main'  # visdom env
    vis_port = 8097  # visdom port
    flag = 'mir'
    
    batch_size = 128
    image_dim = 4096
    hidden_dim = 8192
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 5
    max_epoch = 30

    bit = 64  # hash code length
    lr = 0.00005  # initial learning rate

    device = 'cuda:0'

    # hyper-parameters
    alpha = 10
    gamma = 1
    beta = 1
    mu = 0.00001
    lamb = 1

    margin = 0.4
    dropout = False

    def data(self, flag):
        if flag == 'mir':
            self.dataset = 'flickr'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 10000
            self.data_path = '/home1/ljy/dataset/FLICKR-25K.mat'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)
            
        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                    print('\t{0}: {1}'.format(k, getattr(self, k)))




opt = Default()
