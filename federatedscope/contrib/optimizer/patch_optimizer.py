import torch
import torch.nn as nn

from federatedscope.register import register_optimizer


def get_parameters(model, keys=None, mode="include"):
    if keys is None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield param
    elif mode == "include":
        for name, param in model.named_parameters():
            flag = False
            for key in keys:
                if key in name:
                    flag = True
                    break
            if flag and param.requires_grad:
                yield param
    elif mode == "exclude":
        for name, param in model.named_parameters():
            flag = True
            for key in keys:
                if key in name:
                    flag = False
                    break
            if flag and param.requires_grad:
                yield param
    else:
        raise ValueError("do not support: %s" % mode)


def call_patch_optimizer(model, type, lr, **kwargs):

    if type.startswith('patch') and hasattr(torch.optim, type[5:]):
        # 分类需weight-decay与no-weight-decay的参数
        # no_wd_params, wd_params = [], []
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if ".bn" in name or ".bias" in name:
        #             no_wd_params.append(param)
        #         else:
        #             wd_params.append(param)
        # no_wd_params = nn.ParameterList(no_wd_params)
        # wd_params = nn.ParameterList(wd_params)

        weight_decay = kwargs.pop('weight_decay', 1e-5)
        weight_decay_bn_bias = kwargs.pop('weight_decay_bn_bias', 0)

        params_group = [
            {"params": get_parameters(model, ['bn', 'bias'], mode="exclude"), "weight_decay": weight_decay, 'group_name': 'weight', 'initial_lr': lr},
            {"params": get_parameters(model, ['bn', 'bias'], mode="include"), "weight_decay": weight_decay_bn_bias, 'group_name': 'bn_bias', 'initial_lr': lr},
        ]

        optimizer = getattr(torch.optim, type[5:])(params_group, lr=lr, **kwargs)
        return optimizer


register_optimizer('patch_optimizer', call_patch_optimizer)
