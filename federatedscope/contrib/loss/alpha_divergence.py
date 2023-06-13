from federatedscope.register import register_criterion

from AttentiveNAS.utils.loss_ops import AdaptiveLossSoft


def call_adaptive_alpha_divergence_criterion(type, device):
    if type == 'alpha_divergence':
        criterion = AdaptiveLossSoft().to(device)
        return criterion


register_criterion('alpha_divergence', call_adaptive_alpha_divergence_criterion)
