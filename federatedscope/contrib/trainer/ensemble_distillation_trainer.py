import torch
import copy
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar, lifecycle

from AttentiveNAS.models.attentive_nas_dynamic_model import AttentiveNasDynamicModel

from federatedscope.contrib.auxiliaries.ensemble_related import calculate_ensemble_logits


class DummyContext():
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self):
        pass


# Build your trainer here.
class EnsembleDistillationTrainer(GeneralTorchTrainer):
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

        client_version = config['client_version']

        #  # NOTE(Variant): 若要以以下方式更换，则需放置在__init__之前
        # if not client_version:
        #     self._hook_on_batch_forward = self._hook_on_batch_forward_for_server_model \
        #         if not isinstance(model, AttentiveNasDynamicModel) else self._hook_on_batch_forward_for_supernet

        super(EnsembleDistillationTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        # self._reset_hook_in_trigger()
        self.step_based_epoch_or_batch = 'epoch'

        if not client_version:
            self.ensemble_models = config.ensemble_models  # client models
            # del config.ensemble_models

        self.get_cur_state = config.get_cur_state
        # del config.get_cur_state

        criterion_args = self._cfg.criterion.clone()
        criterion_args.clear_aux_info()
        self.criterion_kwargs = {k: v for k, v in criterion_args.items() if k != "type"}
        # 因specified_cfg由init_cfg经merge得到，因此原有的init_cfg.criterion并不会被覆盖删除，因此在forward时需小心

        if not client_version:
            self.replace_hook_in_train(
                self._hook_on_batch_forward_for_server_model \
                    if not isinstance(self.ctx.model, AttentiveNasDynamicModel) else self._hook_on_batch_forward_for_supernet_debug,  # TODO(Variant): only for debug
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

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:

            round_after = ctx.cfg.train.round_after if hasattr(ctx.cfg.train, 'round_after') else 0  # only for supernet

            ctx.cfg['train'].scheduler['multiplier'] = getattr(ctx, 'num_total_train_batch') \
                if self.step_based_epoch_or_batch == 'batch' else getattr(ctx, 'num_train_epoch')
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
            loss_batch = ctx.criterion(pred, label, **self.criterion_kwargs)
        # if len(label.size()) == 0:
        #     label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    def _hook_on_batch_forward_for_server_model(self, ctx):
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
        x, _ = [_.to(ctx.device) for _ in ctx.data_batch]

        y_logits = calculate_ensemble_logits(x, self.ensemble_models, ctx.cfg.use_amp, ctx.cfg.ensemble_distillation.type)

        with autocast(enabled=ctx.cfg.use_amp):
            prob = ctx.model(x)
            loss_batch = ctx.criterion(prob, y_logits)

        ctx.y_true = CtxVar(y_logits, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(y_logits), LIFECYCLE.BATCH)

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
        x, _ = [_.to(ctx.device) for _ in ctx.data_batch]
        labels = None
        if ctx.cfg.public_data.annotated:  # public labels available
            _, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        ensemble_y_logits = None
        if ctx.cfg.ensemble_distillation.enable:  # ensemble distillation
            ensemble_y_logits = calculate_ensemble_logits(x, self.ensemble_models, ctx.cfg.use_amp,
                                                          ctx.cfg.ensemble_distillation.type)

        def calculate_mixed_loss(pred):
            labels_loss = torch.nn.functional.cross_entropy(pred, labels) if labels is not None else 0
            ensemble_distillation_loss = ctx.criterion(pred, ensemble_y_logits) if ensemble_y_logits is not None else 0
            return labels_loss + ensemble_distillation_loss

        num_arch_training = ctx.cfg.supernet_arch_sampler.num_arch_training
        prob = {arch_id: None for arch_id in range(num_arch_training)}

        loss_batch = []

        inplace_y_logits = None
        if ctx.cfg.inplace_distillation.enable:  # disable ensemble distillation
            ctx.model.sample_min_subnet() if ctx.cfg.inplace_distillation.type == "reverse" else ctx.model.sample_max_subnet()
            ctx.model.set_dropout_rate(0, 0, True)  # NOTE(Variant): derive logits, so disable dropout etc.

            with torch.no_grad() if labels is None and ensemble_y_logits is None else DummyContext():
                with autocast(enabled=ctx.cfg.use_amp):  # TODO(Variant): 原需设定ctx.model.eval()，而supernet不支持eval
                    prob[0] = ctx.model(x)

            loss_batch.append(calculate_mixed_loss(prob[0]))

            with torch.no_grad():
                inplace_y_logits = prob[0].clone().detach()

        # TODO(Variant): 若使能 inplace_distillation，则 true labels 与 ensemble logits 只可被 teacher_subnet 获取(access)，
        #  获取该 teacher subnet 的 inplace_logits 用于监督其他 subnet 的训练。
        # TODO(Variant): 若不使能 inplace_distillation，则 true labels 与 ensemble logits 用于监督所有 subnet 的训练。

        for arch_id in range(1 if ctx.cfg.inplace_distillation.enable else 0, num_arch_training):
            if arch_id == 0:
                ctx.model.sample_min_subnet()
                ctx.model.set_dropout_rate(0, 0, True)
            elif arch_id == num_arch_training - 1:
                if (not ctx.cfg.inplace_distillation.enable) or (ctx.cfg.inplace_distillation.type == "reverse"):
                    ctx.model.sample_max_subnet()
                    # add regularization for the largest subnet
                    ctx.model.set_dropout_rate(ctx.cfg.supernet.drop_out, ctx.cfg.supernet.drop_connect,
                                               drop_connect_only_last_two_stages=ctx.cfg.supernet.drop_connect_only_last_two_stages)
                else:
                    ctx.model.sample_min_subnet()
                    ctx.model.set_dropout_rate(0, 0, True)
            else:
                ctx.model.sample_active_subnet()
                ctx.model.set_dropout_rate(0, 0, True)

            with autocast(enabled=ctx.cfg.use_amp):
                prob[arch_id] = ctx.model(x)
                if ctx.cfg.inplace_distillation.enable:
                    loss = ctx.criterion(prob[arch_id], inplace_y_logits)
                else:
                    loss = calculate_mixed_loss(prob[arch_id])
                loss_batch.append(loss)

        # restore default setting
        ctx.model.sample_max_subnet()
        ctx.model.set_dropout_rate(0, 0, True)

        ctx.y_true = CtxVar(labels if labels is not None else [_.to(ctx.device) for _ in ctx.data_batch][1], LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob[0], LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(sum(loss_batch), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(prob[0]), LIFECYCLE.BATCH)

    def _hook_on_batch_forward_for_supernet_debug(self, ctx):  # TODO(Variant): only for debug
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        num_arch_training = ctx.cfg.supernet_arch_sampler.num_arch_training
        prob = {arch_id: None for arch_id in range(num_arch_training)}

        loss_batch = []

        if ctx.cfg.inplace_distillation.enable:  # disable ensemble distillation
            ctx.model.sample_min_subnet() if ctx.cfg.inplace_distillation.type == "reverse" else ctx.model.sample_max_subnet()
            ctx.model.set_dropout_rate(0, 0, True)  # NOTE(Variant): derive logits, so disable dropout etc.

            with autocast(enabled=ctx.cfg.use_amp):  # TODO(Variant): 原需设定ctx.model.eval()，而supernet不支持eval
                y_logits = ctx.model(x)
                loss = torch.nn.functional.cross_entropy(y_logits, labels, label_smoothing=0.1)  # TODO(Variant): only for debug
            loss_batch.append(loss)
            with torch.no_grad():
                y_logits = y_logits.clone().detach()
            prob[0] = y_logits
        else:
            y_logits = calculate_ensemble_logits(x, self.ensemble_models, ctx.cfg.use_amp,
                                                 ctx.cfg.ensemble_distillation.type)

        for arch_id in range(1 if ctx.cfg.inplace_distillation.enable else 0, num_arch_training):
            if arch_id == 0:
                ctx.model.sample_min_subnet()
                ctx.model.set_dropout_rate(0, 0, True)
            elif arch_id == num_arch_training - 1:
                if (not ctx.cfg.inplace_distillation.enable) or (ctx.cfg.inplace_distillation.type == "reverse"):
                    ctx.model.sample_max_subnet()
                    # add regularization for the largest subnet
                    ctx.model.set_dropout_rate(ctx.cfg.supernet.drop_out, ctx.cfg.supernet.drop_connect,
                                               drop_connect_only_last_two_stages=ctx.cfg.supernet.drop_connect_only_last_two_stages)
                else:
                    ctx.model.sample_min_subnet()
                    ctx.model.set_dropout_rate(0, 0, True)
            else:
                ctx.model.sample_active_subnet()
                ctx.model.set_dropout_rate(0, 0, True)

            with autocast(enabled=ctx.cfg.use_amp):
                prob[arch_id] = ctx.model(x)
                loss = ctx.criterion(prob[arch_id], y_logits)
                loss_batch.append(loss)

        # restore default setting
        ctx.model.sample_max_subnet()
        ctx.model.set_dropout_rate(0, 0, True)

        loss_batch = sum(loss_batch)
        ctx.y_true = CtxVar(y_logits, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob[0], LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(y_logits), LIFECYCLE.BATCH)

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

        if self.step_based_epoch_or_batch == 'batch':
            if ctx.scheduler is not None:
                ctx.scheduler.step()
        else:  # self.step_based_epoch_or_batch == 'epoch'
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
            loss_batch = torch.nn.functional.cross_entropy(prob, label)
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


def call_nas_trainer(trainer_type):
    if trainer_type == 'ensemble_distillation_trainer':
        trainer_builder = EnsembleDistillationTrainer
        return trainer_builder


register_trainer('ensemble_distillation_trainer', call_nas_trainer)
