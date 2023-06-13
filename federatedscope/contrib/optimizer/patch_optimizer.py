import torch
import torch.nn as nn

from federatedscope.register import register_optimizer


def call_patch_optimizer(model, type, lr, **kwargs):

    if type.startswith('patch') and hasattr(torch.optim, type[5:]):
        # 分类需weight-decay与no-weight-decay的参数
        no_wd_params, wd_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if ".bn" in name or ".bias" in name:
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        no_wd_params = nn.ParameterList(no_wd_params)
        wd_params = nn.ParameterList(wd_params)

        weight_decay = kwargs.pop('weight_decay', 1e-5)
        weight_decay_bn_bias = kwargs.pop('weight_decay_bn_bias', 0)

        params_group = [
            {"params": wd_params, "weight_decay": weight_decay, 'group_name': 'weight', 'initial_lr': lr},
            {"params": no_wd_params, "weight_decay": weight_decay_bn_bias, 'group_name': 'bn_bias', 'initial_lr': lr},
        ]

        optimizer = getattr(torch.optim, type[5:])(params_group, lr=lr, **kwargs)
        return optimizer


register_optimizer('patch_optimizer', call_patch_optimizer)
