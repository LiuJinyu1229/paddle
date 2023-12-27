import paddle
import numpy as np

class Dataset(paddle.io.Dataset):
    def __init__(self, opt, images, tags, labels, test=None):
        super(Dataset, self).__init__()
        self.test = test
        all_index = np.arange(tags.shape[0])
        if opt.flag == 'mir':
            query_index = all_index[opt.db_size:]
            training_index = all_index[:opt.training_size]
            db_index = all_index[:opt.db_size]
        else:
            query_index = all_index[:opt.query_size]
            training_index = all_index[opt.query_size: opt.query_size + opt.training_size]
            db_index = all_index[opt.query_size:]

        if test is None:
            train_images = images[training_index]
            train_tags = tags[training_index]
            train_labels = labels[training_index]
            self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = labels[query_index]
            self.db_labels = labels[db_index]
            if test == 'image.query':
                self.images = images[query_index]
            elif test == 'image.db':
                self.images = images[db_index]
            elif test == 'text.query':
                self.tags = tags[query_index]
            elif test == 'text.db':
                self.tags = tags[db_index]
        print(self.test)

    def __getitem__(self, index):
        if self.test is None:
            return (
                index,
                paddle.to_tensor(self.images[index].astype('float32')),
                paddle.to_tensor(self.tags[index].astype('float32')),
                paddle.to_tensor(self.labels[index].astype('float32'))
            )
        elif self.test.startswith('image'):
            return paddle.to_tensor(self.images[index].astype('float32'))
        elif self.test.startswith('text'):
            return paddle.to_tensor(self.tags[index].astype('float32'))

    def __len__(self):
        if self.test is None:
            return len(self.images)
        elif self.test.startswith('image'):
            return len(self.images)
        elif self.test.startswith('text'):
            return len(self.tags)

    def get_labels(self):
        if self.test is None:
            return paddle.to_tensor(self.labels.astype('float32'))
        else:
            return (
                paddle.to_tensor(self.query_labels.astype('float32')),
                paddle.to_tensor(self.db_labels.astype('float32'))
            )