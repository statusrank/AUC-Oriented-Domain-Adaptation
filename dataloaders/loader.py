import argparse
from torch.utils.data import DataLoader
from .dataset import Dataset, StratifiedSampler, StratifiedMultiLabelSampler

def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.
    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to
    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

def get_dataset_names():
    return list(dataset_config())

def dataset_config(name=None):
    # assert name in get_dataset_names()
    config = {
        'NSFW': {
            'lmdb_dir': None,
            'resize_size': 256,
            'is_cen': False,
            'crop_size': 224,
            'batch_size': 32,
            'num_workers': 4,
            'multi_label': False
        },
        'Office31': {
            'lmdb_dir': None,
            'resize_size': 256,
            'is_cen': False,
            'crop_size': 224,
            'batch_size': 32,
            'num_workers': 4,
            'multi_label': False
        },
        'ChestXray': {
            'lmdb_dir': None,
            'resize_size': 256,
            'is_cen': False,
            'crop_size': 224,
            'batch_size': 32,
            'num_workers': 4,
            'multi_label': True
        },
        'tiny-imagenet-200_vs_cifar-10': {
            'lmdb_dir': None,
            'resize_size': 64,
            'is_cen': False,
            'crop_size': 64,
            'batch_size': 256,
            'num_workers': 4,
            'multi_label': False
        },
        'skin': {
            'lmdb_dir': None,
            'resize_size': 256,
            'is_cen': False,
            'crop_size': 224,
            'batch_size': 32,
            'num_workers': 4,
            'multi_label': False
        }
    }

    if name is None:
        return config.keys()

    return config[name]

def load_dataloader(args):
    config = dataset_config(args.data)
    args = vars(args)
    args.update(config)
    args = argparse.Namespace(**args)
    train_source_dataset = Dataset(args, args.source, 'train')
    train_target_dataset = Dataset(args, args.target, 'train')
    test_dataset = Dataset(args, args.target, 'test')
    num_classes = test_dataset.num_classes
    batch_size = args.batch_size
    num_workers = args.num_workers

    # source_labels = set(train_source_dataset.get_labels())
    # target_labels = set(train_target_dataset.get_labels())
    # print("train_source_dataset: ", source_labels)
    # print("train_target_dataset: ", target_labels)

    if args.multi_label:
        source_sampler = StratifiedMultiLabelSampler(
            train_source_dataset.get_labels(),
            batch_size = batch_size
        )
    else:
        source_sampler = StratifiedSampler(train_source_dataset.get_labels(), 
                                      batch_size = batch_size)

    print("=====> number of source domain images: ", len(train_source_dataset))
    print("=====> number of target domain images: ", len(train_target_dataset))
    print("=====> real_batch_size: ", source_sampler.real_batch_size) 
    # target_sampler = StratifiedSampler(train_target_dataset.get_labels(),
    #                                   batch_size = batch_size)

    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=source_sampler.real_batch_size,
        # shuffle=True,
        sampler=source_sampler,
        num_workers=num_workers,
        drop_last=True
    )
    # train_target_loader = DataLoader(
    #     train_target_dataset, 
    #     batch_size=source_sampler.real_batch_size, 
    #     # shuffle=True, 
    #     sampler = target_sampler,
    #     num_workers=num_workers,
    #     drop_last=True
    # )

    train_target_loader = DataLoader(
        train_target_dataset, 
        batch_size=source_sampler.real_batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    if args.da_method == 'AUC_DA' or args.da_method == 'AUCMSourceOnly':
        return train_source_loader, train_target_loader, test_loader, num_classes, train_source_dataset.get_freq_info()
    else:
        return train_source_loader, train_target_loader, test_loader, num_classes
