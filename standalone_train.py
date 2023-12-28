import os
import argparse
import logging
import time
import json

import numpy as np
import torch
import torchvision
import torch.nn as nn
import random

from torch.cuda.amp import GradScaler, autocast

from federatedscope.core.auxiliaries.utils import setup_seed

from federatedscope.contrib.model.attentive_net import call_attentive_net
from AttentiveNAS.models.model_factory import create_model
from train_val import train_one_epoch, evaluate_one_epoch


def get_logger(outdir):
    logger = logging.getLogger()
    logger.setLevel('INFO')

    BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
    DATE_FORMAT = '%m/%d %I:%M:%S %p'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    console_hlr = logging.StreamHandler()
    console_hlr.setFormatter(formatter)
    console_hlr.setLevel('INFO')

    file_hlr = logging.FileHandler(os.path.join(outdir, 'log.txt'))
    file_hlr.setFormatter(formatter)  # level默认，同上面默认设定的DEBUG

    # logger.addHandler(console_hlr)
    logger.addHandler(file_hlr)  # TODO(Variant): logger改写
    return logger


def config_args():
    class ArgsNamespace(argparse.Namespace):
        def get(self, key, default=None):
            return getattr(self, key, default)
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    # parser.add_argument('--use_amp', action='store_true')  # torch.cuda.amp
    parser.add_argument('--gpu', help='gpu available', default='0')  # '0,1'
    parser.add_argument('--data', default='cifar100', choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('--type', default='attentive_subnet')
    parser.add_argument('--drop_out', default=.0, type=float)
    parser.add_argument('--drop_connect', default=.0, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float)
    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.0125, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--manual_seed', default=42, type=int)
    parser.add_argument('--outdir', default='exp-standalone-runs')
    args = parser.parse_args()
    args = ArgsNamespace(**vars(args))

    args.lr = args.batch_size * args.lr / 64
    args.n_classes = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}[args.data.lower()]
    args.smoothing = 0. if args.data.lower() == "cifar10" else 0.1
    args.outdir = args.outdir + f'/{args.data}/'
    os.makedirs(args.outdir, exist_ok=True)
    logger = get_logger(args.outdir)
    logger.info(args)

    args.logger = logger
    return args


def config_random(args):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 更改为相对id

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    args.device = torch.device("cuda:0")

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def standalone_train(args, model, train_loader, val_loader):
    # training related
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    scaler = GradScaler()  # use_amp

    best_acc = 0
    for epoch_i in range(args.epoch):
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, device=args.device)
        loss, acc, _ = evaluate_one_epoch(model, val_loader, use_amp=scaler is not None, device=args.device)
        scheduler.step()

        args.logger.info(f"Epoch [{epoch_i}/{args.epoch}]: Loss {loss:.6f}, Top1.Acc {acc:.5f}")

        # state = {
        #     'state_dict': model.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch_i,
        # }
        #
        # torch.save(state, os.path.join(args.outdir, 'last.pth'))
        if acc > best_acc:
            # torch.save(state, os.path.join(args.outdir, 'best.pth'))
            best_acc = acc
    return {"last_acc": acc, "best_acc": best_acc}


def prepare_dataloader(args):
    # Data
    print('==> Preparing data ...')
    if args.data.lower().startswith("cifar"):
        from federatedscope.contrib.data.cifar import build_cifar_transforms
        train_transforms, val_transforms, test_transforms = build_cifar_transforms(
            args.data.lower(), autoaugment=True, random_erase=False, cutout=False,
        )
    elif args.data.lower() == "tinyimagenet":
        from federatedscope.contrib.data.tinyimagenet import build_tiny_imagenet_transforms
        train_transforms, val_transforms, test_transforms = build_tiny_imagenet_transforms()
    else:
        raise ValueError

    if args.data.lower() == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=test_transforms)
    elif args.data.lower() == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=test_transforms)
    elif args.data.lower() == "tinyimagenet":
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join('data/', 'tiny-imagenet-200', 'train'), transform=train_transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join('data/', 'tiny-imagenet-200', 'val'), transform=test_transforms)
    else:
        raise ValueError

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=6, pin_memory=True, drop_last=False)
    return train_dataloader, val_dataloader


def get_dozens_of_subnets(args):
    from search_related import sample_subnet
    args.type = "attentive_supernet"
    supernet = call_attentive_net(args, input_shape=None).to(args.device)
    args.type = "attentive_subnet"

    sets = []

    subnet_info = sample_subnet(supernet, specs="min")
    sets.append(subnet_info)

    limits = [
        [55, 60], [60, 65], [65, 70], [70, 75],
        [75, 80], [80, 85], [85, 90], [90, 95],
        [95, 100], [100, 105], [105, 110], [110, 115],
        [115, 120], [120, 125], [125, 130], [130, 135], [135, 140]
    ]
    for i, (low, high) in enumerate(limits):
        subnet_info = sample_subnet(supernet, specs="random", flops_limits=(low * 1e6, high * 1e6))
        sets.append(subnet_info)

    subnet_info = sample_subnet(supernet, specs="max")
    sets.append(subnet_info)

    return sets


if __name__ == "__main__":
    args = config_args()
    config_random(args)
    args.device = torch.device("cuda:0")

    train_dataloader, val_dataloader = prepare_dataloader(args)

    # sample dozens of subnets
    subnets_infos = get_dozens_of_subnets(args)

    # Model
    print('==> Building model..')
    for subnet_info in subnets_infos:
        args.arch_cfg = subnet_info['arch_cfg']
        net = call_attentive_net(args, input_shape=None).to(args.device)

        acc_infos = standalone_train(args, net, train_dataloader, val_dataloader)

        subnet_info.update(acc_infos)

    with open(f"{args.outdir}/series_{len(subnets_infos)}subnets_infos.json", "w") as f:
        json.dump(subnets_infos, f, indent=4)






