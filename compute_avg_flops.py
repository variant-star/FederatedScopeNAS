import os
import torch
import thop
import numpy as np
from federatedscope.core.auxiliaries.utils import setup_seed

from federatedscope.core.configs.config import global_cfg, CfgNode, CN
from federatedscope.contrib.auxiliaries.seed_data_builder import get_seed_data

from federatedscope.core.auxiliaries.model_builder import get_model


if __name__ == "__main__":

    wkdir = "exp/FedAgg_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231129090850"  #89.314896
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231129090857"  #92.626688
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar10_lr0.08_lstep0/sub_exp_20231128225610"  #49.920128
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231201221158"  #89.85408
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231201170352"  #91.507456
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar10_lr0.08_lstep0/sub_exp_20231201170402"  #49.920128
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231129094828"  #87.830096
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231129094842"  #90.282736
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar10_lr0.08_lstep0/sub_exp_20231128225614"  #49.920128
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231130150234"  #89.033824
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231130095313"  #92.6973312
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar10_lr0.08_lstep0/sub_exp_20231130095551"  #49.920128

    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231124170739"  #93.37272
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231126174511"  #89.809616
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar100_lr0.08_lstep0/sub_exp_20231124170751"  #50.035328
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231201152553"  #97.563733
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231201090742"  #101.804715
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar100_lr0.08_lstep0/sub_exp_20231201090800"  #50.035328
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231128091114"  #89.28212
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231126162019"  #95.734336
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar100_lr0.08_lstep0/sub_exp_20231126162146"  #50.035328
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231128163519"  #89.28212
    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231128114050"  #95.734336
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar100_lr0.08_lstep0/sub_exp_20231128114101"  #50.035328

    # wkdir = "exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231130144524"  #100.371472
    # wkdir = "exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231130144536"  #105.595232
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_tinyimagenet_lr0.08_lstep0/sub_exp_20231130144822"  #50.163328
    # wkdir = "exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231201151637"  #91.19152
    # wkdir = "exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231201101443"  #102.3150506
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_tinyimagenet_lr0.08_lstep0/sub_exp_20231201101505"  #50.163328
    # wkdir = "exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231130191937"  #94.915696
    # wkdir = "exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231130164735"  #100.0666
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_tinyimagenet_lr0.08_lstep0/sub_exp_20231130164750"  #50.163328
    # wkdir = "exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231204091829"  #95.3318144
    # wkdir = "exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231203141400"  #111.4137408
    # wkdir = "exp/FedTrain_attentive_min_subnet_on_tinyimagenet_lr0.08_lstep0/sub_exp_20231203141416"  #50.163328


    client_cfgs = dict()
    init_cfg = CN.load_cfg(open(os.path.join(wkdir, "configs", "server_config.yaml"), 'r'))
    for cid in range(1, init_cfg.federate.client_num + 1):
        client_cfgs[cid] = CN.load_cfg(open(os.path.join(wkdir, "configs", f"client{cid}_config.yaml"), 'r'))

    # setup_seed(init_cfg.seed)

    supernet_cfg = client_cfgs[1].model.clone()
    supernet_cfg.defrost()
    supernet_cfg.type = "attentive_supernet"
    supernet = get_model(supernet_cfg, local_data=None, backend=init_cfg.backend)

    supernet.sample_max_subnet()
    subnet = supernet.get_active_subnet(preserve_weight=False)

    inputs = torch.randn(1, 3, init_cfg.model.in_resolution, init_cfg.model.in_resolution)  # for thop to compute flops
    flops, params = thop.profile(subnet, inputs=(inputs,), verbose=False)
    print(flops)

    # # data, _ = get_seed_data(config=init_cfg.clone(), client_cfgs=None)
    # inputs = torch.randn(1, 3, 56, 56)  # for thop to compute flops
    #
    # flops_list = []
    # for cid in range(1, init_cfg.federate.client_num + 1):
    #     print(cid)
    #     client_model = get_model(client_cfgs[cid].model, None, backend=client_cfgs[cid].backend)
    #     flops, params = thop.profile(client_model, inputs=(inputs,), verbose=False)
    #     flops_list.append(flops)
    # print(np.mean(flops_list) / 1_000_000)