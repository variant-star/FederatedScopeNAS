from federatedscope.register import register_scheduler

from AttentiveNAS.solver.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR, WarmupLinearDecayLR, ConstantLR


def call_warmup_scheduler(optimizer, reg_type, **kwargs):

    if reg_type == 'warmup_cosine_scheduler':
        assert 'max_iters' in kwargs.keys() and 'warmup_iters' in kwargs.keys() and 'multiplier' in kwargs.keys()

        multiplier = kwargs.pop('multiplier', 1)

        max_iters = max(kwargs.get('max_iters') * multiplier, 1)  # 仅避免multiplier即local_update_step为0的情况
        warmup_iters = kwargs.get('warmup_iters') * multiplier
        warmup_factor = kwargs.get('warmup_factor', 0.001)
        clamp_lr = kwargs.get('clamp_lr', 0.)

        last_epoch = -1
        if 'last_epoch' in kwargs.keys():
            last_epoch = kwargs.get('last_epoch')
            if callable(last_epoch):
                last_epoch = last_epoch()
        last_epoch = max(last_epoch * multiplier, last_epoch)

        return WarmupCosineLR(
            optimizer=optimizer,
            max_iters=max_iters,
            warmup_iters=warmup_iters,
            warmup_factor=warmup_factor,
            warmup_method='linear',
            last_epoch=last_epoch,
            clamp_lr=clamp_lr
        )

    if reg_type == 'warmup_multistep_scheduler':
        return None

    if reg_type == 'warmup_linear_scheduler':
        return None


register_scheduler('warmup_cosine_scheduler', call_warmup_scheduler)
register_scheduler('warmup_multistep_scheduler', call_warmup_scheduler)
register_scheduler('warmup_linear_scheduler', call_warmup_scheduler)