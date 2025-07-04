import os
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from standalone_train import config_args, config_random
from federatedscope.contrib.model.attentive_net import call_attentive_net

# from train_val import train_one_epoch, evaluate_one_epoch

# python conflict.py --data cifar10 --smoothing 0 --batch_size 100 --outdir "PAPER_conflict/temp"


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
    args.in_resolution = {"cifar10": 32, "cifar100": 32, "tinyimagenet": 56}[args.data.lower()]
    args.smoothing = 0. if args.data.lower() == "cifar10" else 0.1
    args.outdir = args.outdir + f'/{args.data}/'
    os.makedirs(args.outdir, exist_ok=True)
    logger = get_logger(args.outdir)
    logger.info(args)

    args.logger = logger
    return args


def prepare_cifar10_dataloader(args, type="seq"):  # seq, cross, crossA, crossB

    # 训练集的转换（包含数据增强）
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    # 测试集的转换（仅基础预处理）
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    def seq_sort_dataset(dataset):
        targets = torch.tensor(dataset.targets)
        sorted_indices = torch.argsort(targets, descending=False)
        return torch.utils.data.Subset(dataset, sorted_indices)

    def cross_sort_dataset(dataset):
        targets = torch.tensor(dataset.targets)
        sorted_indices = torch.argsort(targets, descending=False)

        samples_per_class = len(targets) // 10
        cross_indices = []
        for j in range(samples_per_class):
            for i in range(10):
                cross_indices.append(sorted_indices[i * samples_per_class + j])
        return torch.utils.data.Subset(dataset, cross_indices)

    def crossA_sort_dataset(dataset):
        # 0123401234--5678956789
        targets = torch.tensor(dataset.targets)
        sorted_indices = torch.argsort(targets, descending=False)

        samples_per_class = len(targets) // 10
        cross_indices = []
        for j in range(samples_per_class):
            for i in np.arange(5):
                cross_indices.append(sorted_indices[i * samples_per_class + j])
        for j in range(samples_per_class):
            for i in np.arange(5, 10):
                cross_indices.append(sorted_indices[i * samples_per_class + j])
        return torch.utils.data.Subset(dataset, cross_indices)

    def crossB_sort_dataset(dataset):
        # 5678956789--0123401234
        targets = torch.tensor(dataset.targets)
        sorted_indices = torch.argsort(targets, descending=False)

        samples_per_class = len(targets) // 10
        cross_indices = []
        for j in range(samples_per_class):
            for i in np.arange(5, 10):
                cross_indices.append(sorted_indices[i * samples_per_class + j])
        for j in range(samples_per_class):
            for i in np.arange(5):
                cross_indices.append(sorted_indices[i * samples_per_class + j])
        return torch.utils.data.Subset(dataset, cross_indices)

    # train_dataset = torchvision.datasets.CIFAR10('../data/CIFAR10', train=True, download=False, transform=train_transforms)
    # train_dataset_sorted = sort_dataset_by_class(train_dataset)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset_sorted, batch_size=args.batch_size, shuffle=False,
    #                                                num_workers=6, pin_memory=True, drop_last=True)

    val_dataset = torchvision.datasets.CIFAR10('/home/featurize/data/CIFAR10', train=False, download=True, transform=train_transforms)
    val_dataset_sorted = eval(f"{type}_sort_dataset")(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset_sorted, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=6, pin_memory=True, drop_last=True)
    return None, val_dataloader


def get_flattened_gradients(model, layer_names):
    grads = []
    for name, param in model.named_parameters():
        if any(layer_name == name for layer_name in layer_names):
            if param.grad is not None:
                grads.append(param.grad.view(-1))
    return torch.cat(grads) if grads else torch.tensor([])


def cosine_of_gradient_angle(model1, model2, layer_names=['last_conv.conv.weight', 'classifier.linear.weight']):
    # 获取梯度向量
    grad1 = get_flattened_gradients(model1, layer_names)
    grad2 = get_flattened_gradients(model2, layer_names)

    if grad1.numel() == 0 or grad2.numel() == 0:
        raise ValueError("没有检测到指定层的梯度，请确保: 1) 层名称正确 2) 已执行反向传播")

    # 计算余弦相似度
    cos_sim = cosine_similarity(grad1.unsqueeze(0), grad2.unsqueeze(0), dim=1)

    return cos_sim.item()


def forward_backward_one_step(model, inputs, targets):

    model.zero_grad()
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()
    return


if __name__ == "__main__":
    args = config_args()
    config_random(args)
    args.device = torch.device("cuda:0")

    train_loader, val_loader = prepare_cifar10_dataloader(args, type="cross")
    train_loader1, val_loader1 = prepare_cifar10_dataloader(args, type="crossA")
    train_loader2, val_loader2 = prepare_cifar10_dataloader(args, type="crossB")

    # 创建supernet
    args.type = "attentive_supernet"
    supernet = call_attentive_net(args, input_shape=None).to(args.device)
    args.type = "attentive_subnet"

    # min_subnet_cfg = supernet.sample_min_subnet()
    # min_subnet = supernet.get_active_subnet()
    #
    # max_subnet_cfg = supernet.sample_max_subnet()
    # max_subnet = supernet.get_active_subnet()

    subnet_settings = {'resolution': 32, 'width': [16, 16, 24, 32, 64, 96, 160, 320, 1280], 'depth': [1, 2, 3, 3, 3, 3, 1], 'kernel_size': [3, 3, 5, 5, 5, 5, 3], 'expand_ratio': [1, 3, 3, 3, 3, 6, 6]}
    subnet1_settings = subnet_settings
    subnet2_settings = {'resolution': 32, 'width': [24, 16, 24, 40, 48, 96, 96, 320, 1280], 'depth': [1, 2, 3, 3, 3, 3, 1], 'kernel_size': [3, 3, 3, 3, 3, 3, 3], 'expand_ratio': [1, 3, 3, 3, 3, 6, 6]}
    # max_subnet_settings = {'resolution': 32, 'width': [32, 16, 24, 40, 64, 96, 160, 320, 1280], 'depth': [1, 4, 4, 4, 4, 4, 1], 'kernel_size': [3, 5, 5, 5, 5, 5, 3], 'expand_ratio': [1, 6, 6, 6, 3, 6, 6]}

    supernet.set_active_subnet(**subnet_settings)
    subnet = supernet.get_active_subnet(preserve_weight=True)

    supernet.set_active_subnet(**subnet1_settings)
    subnet1 = supernet.get_active_subnet(preserve_weight=True)

    supernet.set_active_subnet(**subnet2_settings)
    subnet2 = supernet.get_active_subnet(preserve_weight=True)


    iid_2net_angle = []
    non_iid_1net_angle = []
    non_iid_2net_angle = []

    for batch_idx, ((inputs, targets), (inputs1, targets1), (inputs2, targets2)) in enumerate(zip(val_loader, val_loader1, val_loader2)):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        inputs1, targets1 = inputs1.to(args.device), targets1.to(args.device)
        inputs2, targets2 = inputs2.to(args.device), targets2.to(args.device)

        forward_backward_one_step(subnet1, inputs, targets)
        forward_backward_one_step(subnet2, inputs, targets)
        iid_2net_angle.append(cosine_of_gradient_angle(subnet1, subnet2, layer_names=['last_conv.conv.weight', 'classifier.linear.weight']))

        forward_backward_one_step(subnet, inputs1, targets1)
        forward_backward_one_step(subnet1, inputs2, targets2)
        non_iid_1net_angle.append(cosine_of_gradient_angle(subnet, subnet1, layer_names=['last_conv.conv.weight', 'classifier.linear.weight']))

        forward_backward_one_step(subnet1, inputs1, targets1)
        forward_backward_one_step(subnet2, inputs2, targets2)
        non_iid_2net_angle.append(cosine_of_gradient_angle(subnet1, subnet2, layer_names=['last_conv.conv.weight', 'classifier.linear.weight']))


    print(np.mean(iid_2net_angle))
    print(np.mean(non_iid_1net_angle))
    print(np.mean(non_iid_2net_angle))

    # from matplotlib.colors import Normalize
    #
    # df = pd.DataFrame({
    #     'Value': np.concatenate([iid_2net_angle, non_iid_2net_angle]),
    #     'Group': ['IID'] * len(iid_2net_angle) + ['non-IID'] * len(non_iid_2net_angle)
    # })
    #
    # plt.figure(figsize=(4, 3.5))
    # violins = sns.violinplot(x='Group', y='Value', data=df, palette='viridis', alpha=0.5, inner='quartile')
    # plt.ylabel('Gradient Cosine Similarity')
    # plt.xlabel('')
    # plt.tight_layout()
    # plt.savefig("PAPER_conflict/violin.pdf", dpi=600, format="pdf")

    from matplotlib import font_manager

    font_path = "PAPER_conflict/fonts/simsun.ttc"
    font_manager.fontManager.addfont(font_path)
    font_path = "PAPER_conflict/fonts/times.ttf"
    font_manager.fontManager.addfont(font_path)

    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # DejaVu Sans
    plt.rcParams['axes.unicode_minus'] = False

    bins = np.linspace(min(min(iid_2net_angle), min(non_iid_2net_angle)), max(max(iid_2net_angle), max(non_iid_2net_angle)), 80)

    plt.figure(figsize=(5, 3.5))
    plt.hist(iid_2net_angle, bins=bins, density=True, alpha=0.5, color='blue', label='独立同分布（IID）')
    plt.hist(non_iid_2net_angle, bins=bins, density=True, alpha=0.5, color='red', label='非独立同分布（non-IID）')

    sns.kdeplot(iid_2net_angle, color='blue', linewidth=1)
    sns.kdeplot(non_iid_2net_angle, color='red', linewidth=1)

    # plt.xlabel('Gradient Cosine Similarity')
    plt.xlabel('梯度余弦相似度')
    # plt.ylabel('Density')
    plt.ylabel('概率密度')
    plt.legend()
    plt.tight_layout()
    plt.savefig("PAPER_conflict/hist_chinese.pdf", dpi=600, format="pdf")


