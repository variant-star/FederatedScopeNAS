from federatedscope.register import register_criterion

from AttentiveNAS.utils.loss_ops import KLLossSoft


def call_kl_divergence_criterion(type, device):
    if type == 'kl_divergence':
        criterion = KLLossSoft().to(device)
        return criterion


register_criterion('kl_divergence', call_kl_divergence_criterion)
