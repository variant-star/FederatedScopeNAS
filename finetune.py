import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from federatedscope.core.auxiliaries.utils import setup_seed

from federatedscope.core.configs.config import global_cfg, CfgNode, CN
from federatedscope.contrib.auxiliaries.seed_data_builder import get_seed_data

from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

from train_val import train_one_epoch, evaluate_one_epoch

DEVICE = torch.device("cuda:0")
# criterion = F.cross_entropy()
criterion = get_criterion("CrossEntropyLoss", DEVICE)  # TODO(Variant): validate it !


def finetune(cfg, model, dataloader):
    cfg['finetune'].scheduler['multiplier'] = cfg.finetune.local_update_steps
    cfg['finetune'].scheduler['max_iters'] = 1
    cfg['finetune'].scheduler['last_epoch'] = -1

    scaler = GradScaler() if cfg.use_amp else None

    optimizer = get_optimizer(model, **cfg['finetune'].optimizer)
    scheduler = get_scheduler(optimizer, **cfg['finetune'].scheduler)

    _, init_test_acc, _ = evaluate_one_epoch(model, dataloader['test'], use_amp=cfg.use_amp, device=DEVICE)

    torch.optim.swa_utils.update_bn(dataloader['server'], model, device=DEVICE)  # recalibrate_bn
    _, recalibrate_bn_test_acc, _ = evaluate_one_epoch(model, dataloader['test'], use_amp=cfg.use_amp, device=DEVICE)
    print(f"init_test_acc: {init_test_acc}, after_recalibrate_bn_test_acc: {recalibrate_bn_test_acc}")

    for epoch in range(cfg.finetune.local_update_steps):
        _, train_acc, _ = train_one_epoch(model, dataloader['train'], criterion, optimizer, scaler=scaler, device=DEVICE)
        scheduler.step()
        _, test_acc, _ = evaluate_one_epoch(model, dataloader['test'], use_amp=cfg.use_amp, device=DEVICE)

        print(f"[Epoch {epoch+1}/{cfg.finetune.local_update_steps}] "
              f"train_acc: {train_acc:.2f}, test_acc: {test_acc:.2f}")

    return test_acc


if __name__ == "__main__":

    # wkdir = "exp/FedAgg_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231204230848"
    wkdir = "exp/FedTrain_attentive_min_subnet_on_cifar100_lr0.08_lstep0/sub_exp_20231204230833"

    start_point = "retrained_weights"  # "retrained_weights" or "supernet_weights"

    client_cfgs = dict()
    init_cfg = CN.load_cfg(open(os.path.join(wkdir, "configs", "server_config.yaml"), 'r'))
    for cid in range(1, init_cfg.federate.client_num + 1):
        client_cfgs[cid] = CN.load_cfg(open(os.path.join(wkdir, "configs", f"client{cid}_config.yaml"), 'r'))

    setup_seed(init_cfg.seed)

    data, _ = get_seed_data(config=init_cfg.clone(), client_cfgs=None)

    accs = list()

    if start_point == "supernet_weights":
        supernet_cfg = client_cfgs[1].model.clone()
        supernet_cfg.defrost()
        supernet_cfg.type = "attentive_supernet"
        supernet = get_model(supernet_cfg, local_data=None, backend=init_cfg.backend)
        # 加载supernet模型
        saved_state_dict = torch.load(supernet_cfg.pretrain, map_location=torch.device("cpu"))
        if "model" in saved_state_dict:
            saved_state_dict = saved_state_dict["model"]
        if "state_dict" in saved_state_dict:
            saved_state_dict = saved_state_dict["state_dict"]
        supernet.load_state_dict(saved_state_dict, strict=True)

    for cid in range(1, init_cfg.federate.client_num + 1):
        client_model = get_model(client_cfgs[cid].model, data[cid], backend=client_cfgs[cid].backend)

        if start_point == "supernet_weights":
            # supernet中采样模型加载权重
            if client_cfgs[cid].model.type == "attentive_min_subnet":
                supernet.sample_min_subnet()
                pretrained = supernet.get_active_subnet(preserve_weight=True).to(DEVICE)
            elif client_cfgs[cid].model.type == "attentive_subnet":
                assert hasattr(client_cfgs[cid].model, 'arch_cfg')
                supernet.set_active_subnet(**client_cfgs[cid].model.arch_cfg)
                pretrained = supernet.get_active_subnet(preserve_weight=True).to(DEVICE)
            else:  # "attentive_supernet"
                pretrained = supernet

            client_model.load_state_dict(pretrained.state_dict(), strict=True)
            print(f"Client{cid} Loaded pretrained checkpoint from supernet checkpoints.")
        else:
            pretrained_pth = os.path.join(wkdir, "checkpoints", f"client{cid}_model.pth")
            pretrained_state_dict = torch.load(pretrained_pth, map_location="cpu")['model']
            client_model.load_state_dict(pretrained_state_dict, strict=True)
            print(f"Client{cid} Loaded pretrained checkpoint from retrained weights.")

        # client_cfgs[cid].finetune.optimizer.lr = 0.01
        # client_cfgs[cid].finetune.scheduler.warmup_iters = 25
        # client_cfgs[cid].finetune.local_update_steps = 20

        print("-" * 30, f"client{cid} finetune", "-" * 30)
        acc = finetune(client_cfgs[cid], client_model, data[cid])

        accs.append(acc)

    print(f"Client_num: {init_cfg.federate.client_num}")
    print(np.mean(accs))