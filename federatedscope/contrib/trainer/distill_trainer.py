import contextlib

import torch
import copy
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar, lifecycle

from AttentiveNAS.models.attentive_nas_dynamic_model import AttentiveNasDynamicModel

from federatedscope.contrib.auxiliaries.ensemble_related import calculate_ensemble_logits


# Build your trainer here.
class DistillTrainer(GeneralTorchTrainer):
    """
        distillate knowledge from 'ensemble client models' to 'server model' and 'nas supernet'
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None
                 ):

        super(DistillTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        # self.step_based_epoch_or_batch = 'epoch'

        self.ensemble_models = config.ensemble_models  # client models
        # del config.ensemble_models

        self.get_cur_state = config.get_cur_state
        # del config.get_cur_state

        self.replace_hook_in_train(
            self._hook_on_batch_forward_for_supernet,
            'on_batch_forward', target_hook_name='_hook_on_batch_forward')

        self.replace_hook_in_eval(
            self._hook_on_fit_start_init_for_evaluate,  # for supernet, it doesn't support 'eval' mode, so we use "train" mode, but no optimizer and scheduler
            'on_fit_start', target_hook_name='_hook_on_fit_start_init')

        self.replace_hook_in_eval(
            self._hook_on_batch_forward_for_evaluate,  # normal forward, no ensemble distillation
            'on_batch_forward', target_hook_name='_hook_on_batch_forward')

        # prepare mixed precision computation
        self.ctx.scaler = GradScaler() if self.ctx.cfg.use_amp else None

    def _hook_on_fit_start_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.loss_batch_total``            Initialize to 0
            ``ctx.loss_regular_total``          Initialize to 0
            ``ctx.num_samples``                 Initialize to 0
            ``ctx.ys_true``                     Initialize to ``[]``
            ``ctx.ys_prob``                     Initialize to ``[]``
            ==================================  ===========================
        """
        # prepare model and optimizer
        ctx.model.to(ctx.device)

        if ctx.cur_mode == MODE.TRAIN:
            round_after = ctx.cfg.train.round_after if hasattr(ctx.cfg.train, 'round_after') else 0  # only for supernet

            ctx.cfg['train'].scheduler['multiplier'] = getattr(ctx, 'num_total_train_batch') \
                if ctx.cfg.train.batch_or_epoch == 'batch' else getattr(ctx, 'num_train_epoch')
            ctx.cfg['train'].scheduler['max_iters'] = ctx.cfg.federate.total_round_num - round_after
            ctx.cfg['train'].scheduler['last_epoch'] = self.get_cur_state() - round_after

            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)
            # print(ctx.optimizer.param_groups[0]['lr'])

        # TODO: the number of batch and epoch is decided by the current mode
        #  and data split, so the number of batch and epoch should be
        #  initialized at the beginning of the routine

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.y_true``                      Move to `ctx.device`
            ``ctx.y_prob``                      Forward propagation get y_prob
            ``ctx.loss_batch``                  Calculate the loss
            ``ctx.batch_size``                  Get the batch_size
            ==================================  ===========================
        """
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        with autocast(enabled=ctx.cfg.use_amp):
            pred = ctx.model(x)
            loss_batch = ctx.criterion(pred, label)
        # if len(label.size()) == 0:
        #     label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    def _hook_on_batch_forward_for_supernet(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.y_true``                      Move to `ctx.device`
            ``ctx.y_prob``                      Forward propagation get y_prob
            ``ctx.loss_batch``                  Calculate the loss
            ``ctx.batch_size``                  Get the batch_size
            ==================================  ===========================
        """
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

    def _hook_on_batch_backward(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.optimizer``                   Update by gradient
            ``ctx.loss_task``                   Backward propagation
            ``ctx.scheduler``                   Update by gradient
            ==================================  ===========================
        """
        ctx.optimizer.zero_grad()

        if ctx.cfg.use_amp:
            ctx.scaler.scale(ctx.loss_task).backward()
            if ctx.grad_clip > 0:
                ctx.scaler.unscale_(ctx.optimizer)
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
            ctx.scaler.step(ctx.optimizer)
            ctx.scaler.update()
        else:
            ctx.loss_task.backward()
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
            ctx.optimizer.step()

        # if self.step_based_epoch_or_batch == 'batch':
        #     if ctx.scheduler is not None:
        #         ctx.scheduler.step()
        # else:  # self.step_based_epoch_or_batch == 'epoch'
        if ctx.cur_batch_i == getattr(ctx, f"num_{self.ctx.cur_split}_batch") - 1:
            ctx.scheduler.step()

    def _hook_on_fit_end(self, ctx):
        """
        Evaluate metrics.

        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.ys_true``                     Convert to ``numpy.array``
            ``ctx.ys_prob``                     Convert to ``numpy.array``
            ``ctx.monitor``                     Evaluate the results
            ``ctx.eval_metrics``                Get evaluated results from \
            ``ctx.monitor``
            ==================================  ===========================
        """
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
        # 模仿分类任务评估top1 acc或correct等
        if ctx.ys_true.ndim == 2:
            ctx.ys_true = np.argmax(ctx.ys_true, axis=1)  # NOTE(Variant): new added

        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)

    def _hook_on_fit_start_init_for_evaluate(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.loss_batch_total``            Initialize to 0
            ``ctx.loss_regular_total``          Initialize to 0
            ``ctx.num_samples``                 Initialize to 0
            ``ctx.ys_true``                     Initialize to ``[]``
            ``ctx.ys_prob``                     Initialize to ``[]``
            ==================================  ===========================
        """
        # prepare model and optimizer
        ctx.model.to(ctx.device)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward_for_evaluate(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.y_true``                      Move to `ctx.device`
            ``ctx.y_prob``                      Forward propagation get y_prob
            ``ctx.loss_batch``                  Calculate the loss
            ``ctx.batch_size``                  Get the batch_size
            ==================================  ===========================
        """
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]  # NOTE(Variant): we use the true label (only for eval)
        with autocast(enabled=ctx.cfg.use_amp):
            prob = ctx.model(x)
            loss_batch = torch.nn.functional.cross_entropy(prob, label)  # NOTE(Variant): different from _hook_on_batch_forward
        # if len(label.size()) == 0:
        #     label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)  # use the simplest cross_entropy
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    def _hook_on_batch_forward_flop_count(self, ctx):
        pass

    def evaluate(self, target_data_split_name="test", hooks_set=None, mode=MODE.TEST):
        hooks_set = hooks_set or self.hooks_in_eval

        if self.ctx.check_split(target_data_split_name, skip=True):
            with torch.no_grad():
                self._run_routine(mode, hooks_set, target_data_split_name)
        else:
            self.ctx.eval_metrics = dict()

        return self.ctx.eval_metrics


def calculate_mixed_loss(ctx, pred, labels=None, inplace_y_logits=None, ensemble_y_logits=None):
    labels_loss = torch.nn.functional.cross_entropy(
        pred, labels, label_smoothing=0 if pred.size(-1) <= 10 else 0.1
    ) if labels is not None else 0
    inplace_distillation_loss = ctx.criterion(pred, inplace_y_logits / ctx.cfg.inplace_distillation.temperature) \
        if inplace_y_logits is not None else 0  # divergence
    ensemble_distillation_loss = ctx.criterion(pred, ensemble_y_logits / ctx.cfg.ensemble_distillation.temperature) \
        if ensemble_y_logits is not None else 0  # divergence
    return labels_loss + ctx.cfg.inplace_distillation.coef * inplace_distillation_loss + \
        ctx.cfg.ensemble_distillation.coef * ensemble_distillation_loss


def call_distill_trainer(trainer_type):
    if trainer_type == 'distill_trainer':
        trainer_builder = DistillTrainer
        return trainer_builder


register_trainer('distill_trainer', call_distill_trainer)
