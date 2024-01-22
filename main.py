import copy
import os
import sys
import json
import numpy as np
import torch.utils.data.dataset

import thop

os.environ["WANDB_API_KEY"] = '36e1acb021f67c1bdb8c32f4b1e7b74c6287560e'
os.environ['WANDB_MODE'] = 'online'

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.contrib.auxiliaries.seed_data_builder import get_seed_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.model_builder import get_model


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保向下舍入不超过10%
    if new_v < 0.99 * v:  # NOTE(Variant): set 1.0
        new_v += divisor
    return new_v


def distribute_hw_info(size, fmin, fmax):  # size 等同于 每个client的数据规模, min, max 设置为最大最小FLOPs
    k = (fmax - fmin) / (max(size) - min(size))
    b = fmax - k * max(size)

    client_hw_infos = {}
    for cid, x in enumerate(size):
        y = k * x + b
        client_hw_infos[f"client_{cid+1}"] = {"flops_limit": _make_divisible(y, 25) * 1e6}

    return client_hw_infos


def get_clients_hardware_limits(supernet, data, dummy_inputs, client_num):
    supernet.sample_min_subnet()
    min_subnet = supernet.get_active_subnet(preserve_weight=False)
    min_subnet_flops, min_subnet_params = thop.profile(min_subnet, inputs=(dummy_inputs,), verbose=False)
    print(f"[MIN Subnet] FLOPs:{min_subnet_flops/1e6:.2f}M, params:{min_subnet_params/1e6:.2f}M")

    supernet.sample_max_subnet()
    max_subnet = supernet.get_active_subnet(preserve_weight=False)
    max_subnet_flops, max_subnet_params = thop.profile(max_subnet, inputs=(dummy_inputs,), verbose=False)
    print(f"[MAX Subnet] FLOPs:{max_subnet_flops / 1e6:.2f}M, params:{max_subnet_params / 1e6:.2f}M")

    client_hw_infos = distribute_hw_info(
        size=[len(data[cid].train_data) for cid in range(1, client_num+1)],
        fmin=min_subnet_flops/1e6 + 10,
        fmax=max_subnet_flops/1e6 + 1
    )
    return client_hw_infos


def write_data_reports(data, n_classes, client_num, outdir):
    # write data reports
    reports = []
    head_line = f"node####, " + ", ".join([f"cls#{_}" for _ in range(n_classes)]) + ", total_num"
    reports.append(head_line)
    for id in range(client_num + 1):
        for split in ['train', 'val', 'test']:
            subset = getattr(data[id], f'{split}_data')

            indices = None
            if subset is not None:
                while True:
                    if isinstance(subset, torch.utils.data.dataset.Subset):
                        indices = np.array(subset.indices) if indices is None else np.array(subset.indices)[indices]
                        subset = subset.dataset
                    else:
                        targets = np.array(subset.targets)
                        if indices is not None:
                            targets = targets[indices]
                        break
                stat = np.bincount(targets, minlength=n_classes)
            else:
                stat = np.zeros(n_classes).astype(int)
            line = f"{'Server' if id == 0 else 'Client'}" + \
                   f"#{id}#{split}, " + ", ".join([f"{_}" for _ in stat]) + f", {np.sum(stat)}"
            reports.append(line)

    with open(os.path.join(outdir, "data_reports.csv"), "w") as f:
        f.write("\n".join(reports))


def prepare_runner_cfgs():
    init_cfg = global_cfg.clone()
    init_cfg.set_new_allowed(True)  # to receive new config dict

    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    # init_cfg.server_trainer_specified.clear_aux_info()  # TODO(Variant): is it necessary?

    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # TODO(Variant): only support CIFAR10 or CIFAR100 or TinyImageNet
    if init_cfg.data.type.lower().startswith("cifar"):
        n_classes = int(init_cfg.data.type.lower().strip("cifar"))   # 10 or 100
        in_resolution = 32
    elif init_cfg.data.type.lower() == "tinyimagenet":
        n_classes = 200
        in_resolution = 56
    else:
        raise ValueError
    init_cfg.model.n_classes = n_classes
    init_cfg.model.in_resolution = in_resolution

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:  # 若有client_cfg_file
        client_raw_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        client_raw_cfgs.merge_from_list(client_cfg_opt)

        client_cfgs = {}

        client_models_cfg = None
        if client_raw_cfgs["client_models_cfg"] != "":
            with open(client_raw_cfgs["client_models_cfg"], "r") as f:
                client_models_cfg = json.load(f)

        for cid in range(1, init_cfg.federate.client_num+1):
            cur_client_cfg = client_raw_cfgs.clone()
            if client_models_cfg:
                cur_client_cfg.model['arch_cfg'] = client_models_cfg[str(cid)]['arch_cfg']
            client_cfgs.update({f"client_{cid}": cur_client_cfg})
    else:  # 否则，client_cfg等同于server_cfg
        client_cfgs = {f"client_{cid}": init_cfg.clone() for cid in range(1, init_cfg.federate.client_num + 1)}

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, _ = get_seed_data(config=init_cfg.clone(),
                            client_cfgs=client_cfgs)
    # init_cfg.merge_from_other_cfg(modified_cfg)

    write_data_reports(data, n_classes=n_classes, client_num=init_cfg.federate.client_num, outdir=init_cfg.outdir)

    # build temp supernet, and compute client_hw_cfgs
    supernet_cfg = init_cfg.model.clone()
    supernet_cfg.type = "attentive_supernet"
    supernet = get_model(supernet_cfg, None, backend='torch')
    dummy_inputs = torch.randn(1, 3, init_cfg.model.in_resolution, init_cfg.model.in_resolution)
    client_hw_cfgs = get_clients_hardware_limits(supernet, data, dummy_inputs, client_num=init_cfg.federate.client_num)
    del supernet

    # load client_specified_hardware_infos
    for cid in range(1, init_cfg.federate.client_num + 1):
        client_cfgs[f"client_{cid}"].update(client_hw_cfgs[f"client_{cid}"])

    init_cfg['flops_limit'] = float("inf")  # float
    return init_cfg, client_cfgs, data


if __name__ == '__main__':

    init_cfg, node_cfgs, data = prepare_runner_cfgs()

    init_cfg.freeze(inform=False)

    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=node_cfgs)
    _ = runner.run()