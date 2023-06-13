import os
import sys
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
from federatedscope.core.auxiliaries.model_builder import get_model

# if os.environ.get('https_proxy'):
#     del os.environ['https_proxy']
# if os.environ.get('http_proxy'):
#     del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    init_cfg.set_new_allowed(True)  # NOTE(Variant): to receive new config dict

    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    init_cfg.server_trainer_specified.clear_aux_info()
    init_cfg.supernet_trainer_specified.clear_aux_info()

    # with open('federatedscope/server_trainer_specified.yaml', 'r') as f:
    #     server_trainer_specified = yaml.load(f, Loader=yaml.FullLoader)
    # init_cfg.server_trainer_specified = server_trainer_specified
    #
    # with open('federatedscope/supernet_trainer_specified.yaml', 'r') as f:
    #     supernet_trainer_specified = yaml.load(f, Loader=yaml.FullLoader)
    # init_cfg.supernet_trainer_specified = supernet_trainer_specified

    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    init_cfg.model.n_classes = 10 if init_cfg.data.type.lower() == 'cifar10' else 100
    init_cfg.supernet.n_classes = 10 if init_cfg.data.type.lower() == 'cifar10' else 100

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, _ = get_seed_data(config=init_cfg.clone(),
                            client_cfgs=client_cfgs)
    # init_cfg.merge_from_other_cfg(modified_cfg)

    # write data reports
    reports = []
    head_line = f"node####, " + ", ".join([f"cls#{_}" for _ in range(init_cfg.model.n_classes)]) + ", total_num"
    reports.append(head_line)
    for id in range(init_cfg.federate.client_num + 1):
        for split in ['train', 'val', 'test']:
            subset = getattr(data[id], f'{split}_data')

            indices = None
            while True:
                if isinstance(subset, torch.utils.data.dataset.Subset):
                    indices = np.array(subset.indices) if indices is None else np.array(subset.indices)[indices]
                    subset = subset.dataset
                    # targets = np.array(subset.dataset.targets)[subset.indices]
                else:
                    targets = np.array(subset.targets)
                    if indices is not None:
                        targets = targets[indices]
                    break

            stat = np.bincount(targets, minlength=init_cfg.model.n_classes)
            line = f"{'Server' if id == 0 else 'Client'}" + f"#{id}, " + ", ".join([f"{_ / np.sum(stat):.3f}" for _ in stat]) + f", {np.sum(stat)}"
            reports.append(line)

    with open(os.path.join(init_cfg.outdir, "data_reports.csv"), "w") as f:
        f.write("\n".join(reports))
    # DONE!

    init_cfg.freeze()

    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=client_cfgs)
    _ = runner.run()

    # import torch
    #
    # supernet = get_model(init_cfg.supernet, data[0], backend=init_cfg.backend)

