import torch
import copy
import json
import random
from tqdm import trange, tqdm
import thop

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.contrib.auxiliaries.seed_data_builder import get_seed_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.model_builder import get_model

from search_related import random_explore, evolution_search, create_and_evaluate

NUM_CLIENTS = 8


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


def distribute_hw_info(size, fmin=51, fmax=150):  # size 等同于 每个client的数据规模, min, max 设置为最大最小FLOPs
    k = (fmax - fmin) / (max(size) - min(size))
    b = fmax - k * max(size)
    hw_infos = []
    for x in size:
        y = k * x + b
        hw_infos.append(_make_divisible(y, 25))
        print(f"client_hw_limit:{hw_infos[-1]}({y:.2f})")
    return hw_infos


def update_specific_popu_infos(cfg, supernet, data, popu_json=None):

    with open(popu_json, 'r') as f:
        saved_infos = json.load(f)

    # 循环遍历popus中所有架构 ---------------------------------------------------------------------------------------------------
    for i in trange(len(saved_infos)):
        # 构建subnet评估fitness
        saved_infos[i] = create_and_evaluate(cfg, supernet, data, saved_infos[i], finetune=True)

    saved_infos.sort(key=lambda x: x['flops'])
    return saved_infos


if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    init_cfg.set_new_allowed(True)  # NOTE(Variant): to receive new config dict

    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # TODO(Variant): only support CIFAR-10 or CIFAR-100
    n_classes = 10 if init_cfg.data.type.lower() == 'cifar10' else 100
    init_cfg.model.n_classes = n_classes

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_raw_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        client_raw_cfgs.merge_from_list(client_cfg_opt)
        if not client_raw_cfgs.get("personalized_model_cfg"):
            client_cfgs = {f"client_{cid}": client_raw_cfgs for cid in range(1, init_cfg.federate.client_num + 1)}
        else:
            with open(client_raw_cfgs.get("personalized_model_cfg"), "r") as f:
                clients_subnet_cfg = json.load(f)
            client_cfgs = {}
            for cid in range(1, init_cfg.federate.client_num + 1):
                cur_client_cfg = client_raw_cfgs.clone()
                cur_client_cfg.model['arch_cfg'] = clients_subnet_cfg[str(cid)]['arch_cfg']
                client_cfgs.update({f"client_{cid}": cur_client_cfg})
    else:
        client_cfgs = None

    data, _ = get_seed_data(config=init_cfg.clone(),
                            client_cfgs=client_cfgs)
    init_cfg['device'] = torch.device("cuda:0")

    # 创建supernet-------------------------------------------------------------------------------------------------------
    supernet = get_model(init_cfg.model, data[0], backend='torch')

    inputs = torch.randn(1, 3, 32, 32)
    supernet.sample_min_subnet()
    min_subnet = supernet.get_active_subnet(preserve_weight=True)
    min_subnet_flops, min_subnet_params = thop.profile(min_subnet, inputs=(inputs,), verbose=False)
    print(f"[MIN Subnet] FLOPs:{min_subnet_flops/1e6:.2f}M, params:{min_subnet_params/1e6:.2f}M")

    supernet.sample_max_subnet()
    max_subnet = supernet.get_active_subnet(preserve_weight=True)
    max_subnet_flops, max_subnet_params = thop.profile(max_subnet, inputs=(inputs,), verbose=False)
    print(f"[MAX Subnet] FLOPs:{max_subnet_flops / 1e6:.2f}M, params:{max_subnet_params / 1e6:.2f}M")

    # 加载预训练模型
    supernet.to(init_cfg.device)
    sub_path = "exp/nas_fl_lr_on_cifar100_lr0.1_lstep1/sub_exp_20231031114136"

    checkpoint_path = sub_path + "/supernet.pth"
    saved_state_dict = torch.load(checkpoint_path)
    if "model" in saved_state_dict:
        saved_state_dict = saved_state_dict["model"]
    if "state_dict" in saved_state_dict:
        saved_state_dict = saved_state_dict["state_dict"]
    supernet.load_state_dict(saved_state_dict)
    print("load the checkpoint from previous training!!!")

    # ------------------------------------------------------------------------------------------------------------------
    cfg = init_cfg.clone()
    cfg.merge_from_other_cfg(client_cfgs.get('client_1'))  # 改变为client cfg
    # cfg.freeze()
    # ------------------------------------------------------------------------------------------------------------------

    # 根据对search space的探索以及对每个client的数据量设定每个client的FLOPs约束，用以表征每个client个性化的环境------------------

    client_hw_info = distribute_hw_info([len(data[cid].train_data) for cid in range(1, NUM_CLIENTS+1)])
    client_hw_info = {id+1: hw_info*1e6 for id, hw_info in enumerate(client_hw_info)}

    # 逐个client作EA search or random search（基于loss而非accuracy）------------------------------------------------------
    client_best = {}
    for cid in range(1, len(client_hw_info) + 1):
        lotsinfos = evolution_search(cfg, supernet, data[cid], client_hw_info[cid], cid=cid)

        # 保存最好arch的信息
        client_best.update({cid: copy.deepcopy(lotsinfos[0])})
        with open(f"{cfg.outdir}/client_best_infos.json", "w") as f:
            json.dump(client_best, f, indent=4)

# ----------------------------------------------------------------------------------------------------------------------
#     # Servre端作实验
#     lotsinfos = random_explore(cfg, supernet, data[0])  # ignore hardware constraints with random search
#     # 保存轨迹上所有arch的信息
#     with open(f"{cfg.outdir}/server_popu_infos.json", "w") as f:
#         json.dump(lotsinfos, f, indent=4)
#
#     # 根据server端population，更新性能
#     for cid in range(1, len(client_hw_info) + 1):
#         lotsinfos = update_specific_popu_infos(cfg, supernet, data[cid], popu_json=f"{sub_path}/server_popu_infos.json")
#         # 保存轨迹上所有arch的信息
#         with open(f"{sub_path}/client{cid}_serverpopu_infos.json", "w") as f:
#             json.dump(lotsinfos, f, indent=4)
