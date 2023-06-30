from federatedscope.register import register_criterion

from AttentiveNAS.utils.loss_ops import KLLossSoft, AdaptiveLossSoft


def call_divergence_criterion(type, device, **kwargs):
    if type == 'kl_divergence':
        return KLLossSoft(**kwargs).to(device)
    if type == 'alpha_divergence':
        return AdaptiveLossSoft(**kwargs).to(device)


register_criterion('divergence', call_divergence_criterion)
