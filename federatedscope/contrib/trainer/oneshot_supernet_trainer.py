import torch
from torch.cuda.amp import GradScaler, autocast

import copy
import thop

from federatedscope.register import register_trainer
from federatedscope.contrib.trainer.enhance_trainer import EnhanceTrainer

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar, lifecycle


# Build your trainer here.
class OneShotSupernetTrainer(EnhanceTrainer):
    """
        distillate knowledge from 'ensemble client models' to 'nas supernet"
        - note: only can be used in Server.
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None
                 ):

        super(OneShotSupernetTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        self.dummy_input = torch.randn(1, 3, 32, 32).to(device)
        self.flops_limit = self.cfg.flops_limit

    def _hook_on_batch_forward_for_train(self, ctx):
        if ctx.cfg.client_aware_training:
            return self._hook_on_batch_forward_for_train_with_client_aware(ctx)
        else:
            return self._hook_on_batch_forward_for_train_without_client_aware(ctx)

    def _hook_on_batch_forward_for_train_without_client_aware(self, ctx):
        # public_ground_truth
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        prob = {arch_id: None for arch_id in range(ctx.cfg.supernet_arch_sampler.num_arch_training)}
        loss_batch = []

        for arch_id in range(ctx.cfg.supernet_arch_sampler.num_arch_training):
            if arch_id == 0:  # min_subnet supervision setting
                ctx.model.sample_min_subnet()
                ctx.model.set_dropout_rate(0, 0, True)

            elif arch_id < ctx.cfg.supernet_arch_sampler.num_arch_training - 1:
                ctx.model.sample_active_subnet()
                ctx.model.set_dropout_rate(0, 0, True)

            else:  # max_subnet supervision setting
                ctx.model.sample_max_subnet()
                ctx.model.set_dropout_rate(ctx.cfg.model.drop_out, ctx.cfg.model.drop_connect,  # add regularization for the largest subnet
                                           drop_connect_only_last_two_stages=ctx.cfg.model.drop_connect_only_last_two_stages)

            ctx.model.train()
            with autocast(enabled=ctx.cfg.use_amp):
                prob[arch_id] = ctx.model(x)
                loss_batch.append(ctx.criterion(prob[arch_id], labels))

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob[0], LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(sum(loss_batch), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(prob[0]), LIFECYCLE.BATCH)

    def _hook_on_batch_forward_for_train_with_client_aware(self, ctx):
        # public_ground_truth
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        prob = {arch_id: None for arch_id in range(ctx.cfg.supernet_arch_sampler.num_arch_training)}
        loss_batch = []

        for arch_id in range(ctx.cfg.supernet_arch_sampler.num_arch_training):
            while True:
                ctx.model.sample_active_subnet()
                subnet = ctx.model.get_active_subnet(preserve_weight=False)
                flops, _ = thop.profile(subnet, inputs=(self.dummy_input, ), verbose=False)
                if self.flops_limit is not None:
                    if flops < self.flops_limit:
                        break
                else:
                    break

            ctx.model.train()
            with autocast(enabled=ctx.cfg.use_amp):
                prob[arch_id] = ctx.model(x)
                loss_batch.append(ctx.criterion(prob[arch_id], labels))

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob[0], LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(sum(loss_batch), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(prob[0]), LIFECYCLE.BATCH)


def call_oneshot_supernet_trainer(trainer_type):
    if trainer_type == 'oneshot_supernet_trainer':
        trainer_builder = OneShotSupernetTrainer
        return trainer_builder


register_trainer('oneshot_supernet_trainer', call_oneshot_supernet_trainer)
