import torch
from torch.cuda.amp import GradScaler, autocast

from federatedscope.register import register_trainer
from federatedscope.contrib.trainer.enhance_trainer import EnhanceTrainer

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar, lifecycle

from federatedscope.contrib.auxiliaries.ensemble_related import calculate_ensemble_logits


# Build your trainer here.
class EnsembleDistillTrainer(EnhanceTrainer):
    """
        distillate knowledge from 'ensemble client models' to 'single student model'.
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

        super(EnsembleDistillTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        self.ensemble_models = []  # alloc memory for cached ensemble models

    def _hook_on_batch_forward_for_train(self, ctx):
        # public_ground_truth
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        # ensemble_distillation
        ens_logits = calculate_ensemble_logits(
            x, self.ensemble_models, ctx.cfg.use_amp, ctx.cfg.ensemble_distillation.type
        )

        # ensemble training
        ctx.model.train()
        with autocast(enabled=ctx.cfg.use_amp):
            prob = ctx.model(x)
            loss_batch = ctx.criterion(prob, ens_logits, labels if ctx.cfg.public_ground_truth.enable else None)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(prob), LIFECYCLE.BATCH)


def call_ensemble_distill_trainer(trainer_type):
    if trainer_type == 'ensemble_distill_trainer':
        trainer_builder = EnsembleDistillTrainer
        return trainer_builder


register_trainer('ensemble_distill_trainer', call_ensemble_distill_trainer)
