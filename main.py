import copy
import os
import sys
import json
import numpy as np
import torch.utils.data.dataset

os.environ["WANDB_API_KEY"] = '36e1acb021f67c1bdb8c32f4b1e7b74c6287560e'
os.environ['WANDB_MODE'] = 'online'

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

import yaml
import federatedscope.register as register
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.contrib.auxiliaries.seed_data_builder import get_seed_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner


if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    init_cfg.set_new_allowed(True)  # to receive new config dict

    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    # init_cfg.server_trainer_specified.clear_aux_info()  # TODO(Variant): is it necessary?

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
            client_cfgs = {f"client_{cid}": client_raw_cfgs for cid in range(1, init_cfg.federate.client_num+1)}
        else:
            with open(client_raw_cfgs.get("personalized_model_cfg"), "r") as f:
                clients_subnet_cfg = json.load(f)
            client_cfgs = {}
            for cid in range(1, init_cfg.federate.client_num+1):
                cur_client_cfg = client_raw_cfgs.clone()
                cur_client_cfg.model['arch_cfg'] = clients_subnet_cfg[str(cid)]['arch_cfg']
                client_cfgs.update({f"client_{cid}": cur_client_cfg})
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, _ = get_seed_data(config=init_cfg.clone(),
                            client_cfgs=client_cfgs)
    # init_cfg.merge_from_other_cfg(modified_cfg)

    # write data reports
    reports = []
    head_line = f"node####, " + ", ".join([f"cls#{_}" for _ in range(n_classes)]) + ", total_num"
    reports.append(head_line)
    for id in range(init_cfg.federate.client_num + 1):
        for split in ['train', 'val', 'test']:
            subset = getattr(data[id], f'{split}_data')

            indices = None
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
            line = f"{'Server' if id == 0 else 'Client'}" + \
                   f"#{id}#{split}, " + ", ".join([f"{_}" for _ in stat]) + f", {np.sum(stat)}"
            reports.append(line)

    with open(os.path.join(init_cfg.outdir, "data_reports.csv"), "w") as f:
        f.write("\n".join(reports))

    init_cfg.freeze(inform=False)

    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=client_cfgs)
    _ = runner.run()

