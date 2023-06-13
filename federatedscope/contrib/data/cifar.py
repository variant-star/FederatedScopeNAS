from ..transform_ops.ops import Cutout, RandomErase
from ..transform_ops.autoaugment import CIFAR10Policy
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from federatedscope.register import register_data
from federatedscope.core.data.base_translator import BaseDataTranslator


def build_transforms(dataset, autoaugment=False, random_erase=False, cutout=False):
    if dataset.lower() == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # cifar10
    elif dataset.lower() == "cifar100":
        normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # cifar100
    else:
        raise ValueError

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

    train_transforms.transforms.append(transforms.ToTensor())
    train_transforms.transforms.append(normalize)

    if cutout:
        cutout_length = 8
        train_transforms.transforms.append(Cutout(1, cutout_length))

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

    train_transforms, val_transforms, test_transforms = build_transforms(
        config.data.type.lower(),
        autoaugment=getattr(config.data, "autoaugment", True),
        random_erase=getattr(config.data, "random_erase", True),
        cutout=getattr(config.data, "cutout", True),
    )

    if config.data.type.lower() == "cifar10":
        raw_train_dataset = datasets.CIFAR10(config.data.root, train=True, download=True, transform=train_transforms)
        raw_test_dataset = datasets.CIFAR10(config.data.root, train=False, download=True, transform=test_transforms)
    else:
        raw_train_dataset = datasets.CIFAR100(config.data.root, train=True, download=True, transform=train_transforms)
        raw_test_dataset = datasets.CIFAR100(config.data.root, train=False, download=True, transform=test_transforms)
    assert len(raw_train_dataset) == 50_000 and len(raw_test_dataset) == 10_000

    translator = BaseDataTranslator(config, client_cfgs)

    # 构建server端unlabeled data，剩余部分用于划分给所有clients
    all_clients_train_dataset, _, server_unlabeled_dataset = translator.split_train_val_test(raw_train_dataset, config)
    fs_data = translator((all_clients_train_dataset, raw_test_dataset, []))

    from federatedscope.core.data import ClientData
    from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
    from copy import deepcopy
    fs_data[0] = ClientData(config, train=server_unlabeled_dataset, val=raw_test_dataset, test=raw_test_dataset)  #
    for client_id in range(1, config.federate.client_num + 1):
        fs_data[client_id].test_data = deepcopy(fs_data[0].test_data)
        # fs_data[client_id].setup(config)  # NOTE(Variant): config not change, so setup() func not work.
        fs_data[client_id]['test'] = get_dataloader(fs_data[client_id].test_data, config, 'test')

    return fs_data, config.clone()

def call_cifar_data(config, client_cfgs=None):
    if config.data.type.lower().startswith("cifar"):
        data, modified_config = load_cifar_data(config, client_cfgs)
        return data, modified_config


register_data("cifar", call_cifar_data)
