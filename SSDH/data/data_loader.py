import paddle
from paddle.io import Dataset, DataLoader
import h5py
from PIL import Image
from paddle.vision.transforms import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DatasetProcess(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, data_path, mode, transform=None):
        self.transform = transform
        self.mode = mode
        h5f = h5py.File(data_path, 'r')

        if mode == 'train':
            self.imgs = h5f["train_data"]
            self.labels = h5f["train_L"]
        elif mode == 'test':
            self.imgs = h5f["test_data"]
            self.labels = h5f["test_L"]
        elif mode == 'database':
            self.imgs = h5f["data_set"]
            self.labels = h5f["dataset_L"]
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img_data = self.imgs[index]
        label = self.labels[index]
        img = Image.fromarray(img_data)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index
    
    def __len__(self):
        return self.imgs.shape[0]

def load_data(dataset, data_path, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        data_path(str): Path of dataset.
        num_workers(int): Number of loading data threads.

    Returns
        test_dataloader, train_dataloader, database_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    test_dataset = DatasetProcess(data_path, 'test', test_transform)
    train_dataset = DatasetProcess(data_path, 'train', train_transform)
    database_dataset = DatasetProcess(data_path, 'database', test_transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    database_dataloader = DataLoader(
        database_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    train_label = train_dataset.labels
    test_label = test_dataset.labels
    database_label = database_dataset.labels

    return test_dataloader, train_dataloader, database_dataloader, test_label, train_label, database_label
