import os
from ..transform_ops.ops import Cutout, RandomErase
from ..transform_ops.autoaugment import CIFAR10Policy
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from federatedscope.register import register_data
from federatedscope.core.data.base_translator import BaseDataTranslator

from federatedscope.core.data import ClientData
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader

import copy


def build_cifar_transforms(dataset, autoaugment=False, random_erase=False, cutout=False):
    if dataset.lower() == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # cifar10
    else:  # dataset.lower() == "cifar100":
        normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # cifar100

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    if random_erase:
        train_transforms.transforms.append(RandomErase(prob=0.5, sl=0.02, sh=0.4, r=0.3))
    # https://github1s.com/pprp/PyTorch-CIFAR-Model-Hub/blob/HEAD/lib/dataset/transforms/build.py 中, random_erase与autoaugment不同时使用

    if autoaugment:
        train_transforms.transforms.append(CIFAR10Policy())
    else:
        train_transforms.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))

    if cutout:
        train_transforms.transforms.append(Cutout(1, length=8))

    train_transforms.transforms.append(transforms.ToTensor())
    train_transforms.transforms.append(normalize)

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, val_transforms, test_transforms


def load_cifar_data(config, client_cfgs=None):

    train_transforms, val_transforms, test_transforms = build_cifar_transforms(
        config.data.type.lower(),
        autoaugment=getattr(config.data, "autoaugment", True),
        random_erase=getattr(config.data, "random_erase", False),
        cutout=getattr(config.data, "cutout", False),
    )
    raw_train_dataset = eval(f"datasets.{config.data.type.upper()}")(config.data.root, train=True, download=True,
                                                                     transform=train_transforms)
    raw_test_dataset = eval(f"datasets.{config.data.type.upper()}")(config.data.root, train=False, download=True,
                                                                    transform=test_transforms)
    assert len(raw_train_dataset) == 50_000 and len(raw_test_dataset) == 10_000

    translator = BaseDataTranslator(config, client_cfgs)

    # 构建server端unlabeled data，剩余部分用于划分给所有clients
    all_clients_train_dataset, all_clients_val_dataset, server_dataset = translator.split_train_val_test(raw_train_dataset, config)
    # copy.deepcopy，避免修改transform时全部修改
    all_clients_train_dataset = copy.deepcopy(all_clients_train_dataset)
    all_clients_val_dataset = copy.deepcopy(all_clients_val_dataset)
    server_dataset = copy.deepcopy(server_dataset)

    all_clients_val_dataset.dataset.transform = test_transforms

    fs_data = translator((all_clients_train_dataset, all_clients_val_dataset, raw_test_dataset))  # raw_test_dataset will be divided into #client_num parts

    server_val_dataset = copy.deepcopy(server_dataset)
    server_val_dataset.dataset.transform = test_transforms

    bn_recalibration_dataset = dict()
    bn_recalibration_dataset['train_version'] = copy.deepcopy(server_dataset)
    bn_recalibration_dataset['train_version'].dataset.transform = train_transforms  # 若改为 test_transform， calibrate bn 性能较差
    # bn_recalibration_dataset['test_version'] = copy.deepcopy(server_dataset)
    # bn_recalibration_dataset['test_version'].dataset.transform = test_transforms

    fs_data[0] = ClientData(config, train=server_dataset, val=server_val_dataset, test=raw_test_dataset)

    for client_id in range(config.federate.client_num + 1):  # server is also included.
        # NOTE(Variant): 额外添加bn_recalibration_dataset
        # the bn_recalibration_dataset(server data): train_transform_version
        fs_data[client_id].server_data = copy.deepcopy(bn_recalibration_dataset['train_version'])
        fs_data[client_id]['server'] = get_dataloader(fs_data[client_id].server_data, config, 'server')
        # the bn_recalibration_dataset(server data): test_transform_version
        # fs_data[client_id].server_data = copy.deepcopy(bn_recalibration_dataset['test_version'])
        # fs_data[client_id]['server_test'] = get_dataloader(fs_data[client_id].server_test_data, config, 'server')

    return fs_data, config.clone()


def call_cifar_data(config, client_cfgs=None):  # CIFAR10, CIFAR100, TinyImageNet
    if config.data.type.lower().startswith("cifar"):
        data, modified_config = load_cifar_data(config, client_cfgs)
        return data, modified_config


register_data("cifar_datasets", call_cifar_data)