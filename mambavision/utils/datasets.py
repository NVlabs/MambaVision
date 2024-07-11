"""Code for getting the data loaders."""

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch._utils import _accumulate
from timm.data import IterableImageDataset, ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset


def get_loaders(args, mode='eval', dataset=None):
    """Get data loaders for required dataset."""
    if dataset is None:
        dataset = args.dataset
    if dataset == 'imagenet':
        return get_imagenet_loader(args, mode)
    else:
        if mode == 'search':
            return get_loaders_search(args)
        elif mode == 'eval':
            return get_loaders_eval(dataset, args)


class Subset_imagenet(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset , indices) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = None

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)


def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar100':
        num_classes = 100
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(
            root=args.data, train=False, download=True, transform=valid_transform)

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)

        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=16)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        sampler=valid_sampler, pin_memory=True, num_workers=16)

    return train_queue, valid_queue, num_classes


def get_loaders_search(args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if args.dataset == 'cifar10':
        num_classes = 10
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    print('Found %d samples' % (num_train))
    sub_num_train = int(np.floor(args.train_portion * num_train))
    sub_num_valid = num_train - sub_num_train

    sub_train_data, sub_valid_data = my_random_split(
        train_data, [sub_num_train, sub_num_valid], seed=0)
    print('Train: Split into %d samples' % (len(sub_train_data)))
    print('Valid: Split into %d samples' % (len(sub_valid_data)))

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            sub_train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            sub_valid_data)

    train_queue = torch.utils.data.DataLoader(
        sub_train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=16, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        sub_valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=16, drop_last=True)

    return train_queue, valid_queue, num_classes

################################################################################
# ImageNet
################################################################################
def get_imagenet_loader(args, mode='eval', testdir = ""):
    """Get train/val for imagenet."""
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    print("verify testing path")
    if len(testdir) < 2:
        testdir = os.path.join("../ImageNetV2/", 'test')
        # print("\n\n\n loading imagenet v2 \n\n\n")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    downscale = 1
    val_transform = transforms.Compose([
        transforms.Resize(args.resize//downscale),
        transforms.CenterCrop(args.resolution//downscale),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.resolution//downscale),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if mode == 'eval':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=train_transform)
            valid_data = imagenet_lmdb_dataset(
                validdir, transform=val_transform)
        else:
            train_data = dset.ImageFolder(traindir, transform=train_transform)
            valid_data = dset.ImageFolder(validdir, transform=val_transform)

        train_sampler, valid_sampler = None, None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data)

            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_data)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            pin_memory=True, num_workers=16, sampler=train_sampler, drop_last=True)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=(valid_sampler is None),
            pin_memory=True, num_workers=16, sampler=valid_sampler)
    elif mode == 'search':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data = dset.ImageFolder(traindir, val_transform)

        num_train = len(train_data)
        print('Found %d samples' % (num_train))
        sub_num_train = int(np.floor(args.train_portion * num_train))
        sub_num_valid = num_train - sub_num_train

        sub_train_data, sub_valid_data = my_random_split(
            train_data, [sub_num_train, sub_num_valid], seed=0)
        print('Train: Split into %d samples' % (len(sub_train_data)))
        print('Valid: Split into %d samples' % (len(sub_valid_data)))

        train_sampler, valid_sampler = None, None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                sub_train_data)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                sub_valid_data)

        train_queue = torch.utils.data.DataLoader(
            sub_train_data, batch_size=args.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            pin_memory=True, num_workers=16, drop_last=True)

        valid_queue = torch.utils.data.DataLoader(
            sub_valid_data, batch_size=args.batch_size,
            sampler=valid_sampler, shuffle=(valid_sampler is None),
            pin_memory=True, num_workers=16, drop_last=False)


    elif mode == 'timm':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=None)
            valid_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data =  ImageDataset(traindir)
            valid_data = dset.ImageFolder(traindir, transform=val_transform)

        train_interpolation = 'bicubic'
        train_queue = create_loader(
            train_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=0.2,
            re_mode="pixel",
            re_count=1,
            re_split=False,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333333333],
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5",
            num_aug_splits=0,
            interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=16,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            use_multi_epochs_loader=False
        )

        num_train = len(valid_data)
        print('Found %d samples' % (num_train))
        sub_num_train = int(np.floor(args.train_portion * num_train))
        sub_num_valid = num_train - sub_num_train

        _, sub_valid_data = my_random_split(
            valid_data, [sub_num_train, sub_num_valid], seed=0)

        print('Valid: Split into %d samples' % (len(sub_valid_data)))

        train_sampler, valid_sampler = None, None
        if args.distributed:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                sub_valid_data)

        valid_queue = torch.utils.data.DataLoader(
            sub_valid_data, batch_size=args.batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler, pin_memory=True, num_workers=16, drop_last=False)

    elif mode == 'timm2':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=None)
            valid_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data =  ImageDataset(traindir)

        valid_data = ImageDataset(testdir)

        train_interpolation = "bicubic"
        train_queue = create_loader(
            train_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=0.2,
            re_mode="pixel",
            re_count=1,
            re_split=False,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333333333],
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5",
            num_aug_splits=0,
            # interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=16,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            use_multi_epochs_loader=False
        )
        valid_queue = create_loader(
            valid_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=16,
            distributed=args.distributed,
            crop_pct=0.875,
            color_jitter=0.4,
            pin_memory=False,
        )

    elif mode == 'timm3':
        # with test set from ImageNetV2 test split
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=None)
            valid_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data = ImageDataset(traindir)

        valid_data = ImageDataset(testdir)
        # valid_data = ImageDataset(traindir)

        train_interpolation = 'bicubic'
        train_queue = create_loader(
            train_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=0.2,
            re_mode="pixel",
            re_count=1,
            re_split=False,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333333333],
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5",
            num_aug_splits=0,
            interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            # num_workers=16,
            num_workers=8,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            use_multi_epochs_loader=False
        )

        valid_queue = create_loader(
            valid_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size * 4,
            is_training=True,
            use_prefetcher=True,
            interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            # num_workers=16,
            num_workers=8,
            distributed=args.distributed,
            crop_pct=0.875,
            color_jitter=0.0,
            pin_memory=False,
        )

    return train_queue, valid_queue, 1000

################################################################################


def my_random_split(dataset, lengths, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(sum(lengths), generator=g)
    return [Subset_imagenet(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
################################################################################


def my_random_split_perc(dataset, percent_train, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        percent_train (float): portion of the dataset to be used for training
    """

    num_train = len(dataset)
    print('Found %d samples' % (num_train))
    sub_num_train = int(np.floor(percent_train * num_train))
    sub_num_valid = num_train - sub_num_train
    dataset_train, dataset_validation = my_random_split(dataset, [sub_num_train, sub_num_valid], seed=seed)
    print('Train: Split into %d samples' % (len(dataset)))


    return [dataset_train, dataset_validation]


################################################################################
# ImageNet - LMDB
################################################################################

import io
import os
try:
    import lmdb
except:
    pass
import torch
from torchvision import datasets
from PIL import Image


def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None,
        loader=lmdb_loader):
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)
    return data_set


if __name__ == '__main__':
    import torch.distributed as dist
    import argparse
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser('Cell search')
    args = parser.parse_args()
    args.data = '/data/datasets/imagenet_lmdb/'
    args.train_portion = 0.9
    args.batch_size = 48
    args.seed = 1
    args.local_rank = 0

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    q1, q2, _ = get_imagenet_loader(args, mode='search')

    iterator = iter(q1)
    input_search, target_search = next(iterator)

    print(len(q1), len(q2))
    ind = 0
    for batch, target in q1:
        """
        img = batch[0].numpy().transpose(1, 2, 0)[:, :, 0]
        plt.imshow(img)
        plt.show()
        plt.pause(1.)
        """
        if ind % 100 == 0:
            print(ind)
        ind += 1

    t1, t2, _ = get_imagenet_loader(args, mode='eval')
    print(len(t1), len(t2))
    for batch, target in t1:
        img = batch[0].numpy().transpose(1, 2, 0)[:, :, 0]
        plt.imshow(img)
        plt.show()
        plt.pause(1.)
        break
