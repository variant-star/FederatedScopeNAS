import os
import copy
import json
import torch
import random
import numpy as np
from tqdm import trange, tqdm

from federatedscope.core.auxiliaries.utils import setup_seed

from federatedscope.core.configs.config import global_cfg, CfgNode, CN
from federatedscope.contrib.auxiliaries.seed_data_builder import get_seed_data
from federatedscope.core.auxiliaries.model_builder import get_model

from search_related import random_explore, evolution_search, create_and_evaluate


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
    # from main import prepare_runner_cfgs
    #
    # init_cfg, client_cfgs, data = prepare_runner_cfgs()
    #
    # for k, v in copy.deepcopy(client_cfgs).items():
    #     client_specific_cfg = init_cfg.clone()
    #     client_specific_cfg.merge_from_other_cfg(client_cfgs[k])
    #     client_cfgs[k] = client_specific_cfg

    wkdir = "exp/OneShot_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20240119140438"
    DEVICE = torch.device("cuda:0")

    client_cfgs = dict()
    init_cfg = CN.load_cfg(open(os.path.join(wkdir, "configs", "server_config.yaml"), 'r'))
    for cid in range(1, init_cfg.federate.client_num + 1):
        client_cfgs[f"client_{cid}"] = CN.load_cfg(open(os.path.join(wkdir, "configs", f"client{cid}_config.yaml"), 'r'))

    setup_seed(init_cfg.seed)

    data, _ = get_seed_data(config=init_cfg.clone(), client_cfgs=None)

    supernet = get_model(init_cfg.model, None, backend="torch").to(DEVICE)

    saved_state_dict = torch.load(os.path.join(wkdir, "checkpoints", "supernet.pth"))["model"]
    supernet.load_state_dict(saved_state_dict, strict=True)
    print("load the checkpoint from previous NAS supernet training!!!")

    # -----------------------reset seed---------------------------------------------------------------------------------
    # np.random.seed(0)
    # random.seed(0)

    # 逐个client作EA search or random search（基于loss而非accuracy）------------------------------------------------------
    client_best = {}
    for cid in range(1, init_cfg.federate.client_num + 1):
        saved_infos = evolution_search(client_cfgs[f"client_{cid}"], supernet, data[cid],
                                       client_cfgs[f"client_{cid}"]['flops_limit'], cid=cid, device=DEVICE)
        client_best.update({cid: copy.deepcopy(saved_infos[0])})
        with open(f"{init_cfg.outdir}/client_best_infos.json", "w") as f:
            json.dump(client_best, f, indent=4)

    # # 逐个client作随机搜索（仅单次随机搜索）
    # from search_related import sample_subnet
    # dummy_inputs = torch.randn(1, 3, init_cfg.model.in_resolution, init_cfg.model.in_resolution).to(DEVICE)
    # client_random = {}
    # for cid in range(1, init_cfg.federate.client_num + 1):
    #     print(f"Client_id: {cid}")
    #     subnet_info = sample_subnet(supernet, specs="random",
    #                                 flops_limits=(client_cfgs[f"client_{cid}"]['flops_limit'] - 20 * 1e6,
    #                                               client_cfgs[f"client_{cid}"]['flops_limit']),
    #                                 dummy_inputs=dummy_inputs,
    #                                 device=DEVICE)
    #     client_random.update({cid: copy.deepcopy(subnet_info)})
    #     with open(f"{init_cfg.outdir}/client_random_infos.json", "w") as f:
    #         json.dump(client_random, f, indent=4)

# 消融实验---------------------------------------------------------------------------------------------------------------
#     # Server端作实验
#     lotsinfos = random_explore(client_cfgs["client_1"], supernet, data[0], finetune=False, max_trials=1000, device=?)  # ignore hardware constraints with random search
#     # 保存轨迹上所有arch的信息
#     with open(f"{init_cfg.outdir}/server_popu_infos.json", "w") as f:
#         json.dump(lotsinfos, f, indent=4)

    # # 根据server端population，更新性能
    # for cid in range(1, init_cfg.federate.client_num + 1):
    #     lotsinfos = update_specific_popu_infos(client_cfgs[f"client_{cid}"], supernet, data[cid], popu_json=f"{sub_path}/server_popu_infos.json")
    #     # 保存轨迹上所有arch的信息
    #     with open(f"{init_cfg.outdir}/client{cid}_serverpopu_infos.json", "w") as f:
    #         json.dump(lotsinfos, f, indent=4)
