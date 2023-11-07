import torch
from torch.cuda.amp import GradScaler, autocast

from federatedscope.register import register_trainer
from federatedscope.contrib.trainer.enhance_trainer import EnhanceTrainer

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar, lifecycle

from federatedscope.contrib.auxiliaries.ensemble_related import calculate_ensemble_logits


# Build your trainer here.
class DistillSupernetTrainer(EnhanceTrainer):
    """
        distillate knowledge from 'ensemble client models' to 'nas supernet"
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None
                 ):

        super(DistillSupernetTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        self.replace_hook_in_train(
            self._hook_on_batch_forward_for_supernet,
            'on_batch_forward', target_hook_name='_hook_on_batch_forward')

    def _hook_on_batch_forward_for_supernet(self, ctx):
        # public_ground_truth
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        # ensemble_distillation
        ens_logits = calculate_ensemble_logits(
            x, self.ensemble_models, ctx.cfg.use_amp, ctx.cfg.ensemble_distillation.type
        ) if ctx.cfg.ensemble_distillation.enable else None

        # inplace_distillation
        inplace_logits = None
        if ctx.cfg.inplace_distillation.enable:
            ctx.model.sample_min_subnet() if ctx.cfg.inplace_distillation.type == "reverse" else ctx.model.sample_max_subnet()
            ctx.model.set_dropout_rate(0, 0, True)  # NOTE(Variant): derive logits, so disable dropout etc.

            ctx.model.train()  # NOTE(Variant): IMPORTANT! MUST BE IN TRAIN MODE!
            with torch.no_grad():
                with autocast(enabled=ctx.cfg.use_amp):
                    pred = ctx.model(x)
                    inplace_logits = pred.clone().detach()

        # supernet training
        prob = {arch_id: None for arch_id in range(ctx.cfg.supernet_arch_sampler.num_arch_training)}
        loss_batch = []

        for arch_id in range(ctx.cfg.supernet_arch_sampler.num_arch_training):
            if arch_id == 0:  # min_subnet supervision setting
                ctx.model.sample_min_subnet()
                ctx.model.set_dropout_rate(0, 0, True)
                enable_labels = ctx.cfg.public_ground_truth.enable and "min" in ctx.cfg.public_ground_truth.include
                enable_inplace = ctx.cfg.inplace_distillation.enable and ctx.cfg.inplace_distillation.type == "normal"
                enable_ens = ctx.cfg.ensemble_distillation.enable and "min" in ctx.cfg.ensemble_distillation.include

            elif arch_id < ctx.cfg.supernet_arch_sampler.num_arch_training - 1:
                ctx.model.sample_active_subnet()
                ctx.model.set_dropout_rate(0, 0, True)
                enable_labels = ctx.cfg.public_ground_truth.enable and "non-extreme" in ctx.cfg.public_ground_truth.include
                enable_inplace = ctx.cfg.inplace_distillation.enable
                enable_ens = ctx.cfg.ensemble_distillation.enable and "non-extreme" in ctx.cfg.ensemble_distillation.include

            else:  # max_subnet supervision setting
                ctx.model.sample_max_subnet()
                ctx.model.set_dropout_rate(ctx.cfg.model.drop_out, ctx.cfg.model.drop_connect, # add regularization for the largest subnet
                                           drop_connect_only_last_two_stages=ctx.cfg.model.drop_connect_only_last_two_stages)
                enable_labels = ctx.cfg.public_ground_truth.enable and "max" in ctx.cfg.public_ground_truth.include
                enable_inplace = ctx.cfg.inplace_distillation.enable and ctx.cfg.inplace_distillation.type == "reverse"
                enable_ens = ctx.cfg.ensemble_distillation.enable and "max" in ctx.cfg.ensemble_distillation.include

            # print(f"arch_id({arch_id}): labels({enable_labels}), inplace({enable_inplace}), ensemble({enable_ens})")

            ctx.model.train()
            with autocast(enabled=ctx.cfg.use_amp):
                prob[arch_id] = ctx.model(x)

                loss_batch.append(calculate_mixed_loss(
                    ctx, prob[arch_id],
                    labels=labels if enable_labels else None,
                    inplace_y_logits=inplace_logits if enable_inplace else None,
                    ensemble_y_logits=ens_logits if enable_ens else None
                ))

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob[0], LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(sum(loss_batch), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(prob[0]), LIFECYCLE.BATCH)


def calculate_mixed_loss(ctx, pred, labels=None, inplace_y_logits=None, ensemble_y_logits=None):
    labels_loss = torch.nn.functional.cross_entropy(
        pred, labels, label_smoothing=0 if pred.size(-1) <= 10 else 0.1
    ) if labels is not None else 0
    inplace_distillation_loss = ctx.criterion(pred, inplace_y_logits) \
        if inplace_y_logits is not None else 0  # divergence
    ensemble_distillation_loss = ctx.criterion(pred, ensemble_y_logits) \
        if ensemble_y_logits is not None else 0  # divergence
    return labels_loss + ctx.cfg.inplace_distillation.coef * inplace_distillation_loss + \
        ctx.cfg.ensemble_distillation.coef * ensemble_distillation_loss


def call_distill_supernet_trainer(trainer_type):
    if trainer_type == 'distill_supernet_trainer':
        trainer_builder = DistillSupernetTrainer
        return trainer_builder


register_trainer('distill_supernet_trainer', call_distill_supernet_trainer)
