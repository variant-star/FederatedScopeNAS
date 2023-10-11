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

        super(EnsembleDistillationTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        # self.step_based_epoch_or_batch = 'epoch'

        if not client_version:
            self.ensemble_models = config.ensemble_models  # client models
            # del config.ensemble_models

        self.get_cur_state = config.get_cur_state
        # del config.get_cur_state

        if not client_version:
            self.replace_hook_in_train(
                self._hook_on_batch_forward_for_server_model \
                    if not isinstance(self.ctx.model, AttentiveNasDynamicModel) else self._hook_on_batch_forward_for_supernet,  # TODO(Variant): only for debug
                'on_batch_forward', target_hook_name='_hook_on_batch_forward')
            self.replace_hook_in_eval(
                self._hook_on_fit_start_init_for_evaluate,  # for supernet, it doesn't support 'eval' mode, so we use "train" mode, but no optimizer and scheduler
                'on_fit_start', target_hook_name='_hook_on_fit_start_init')

        self.replace_hook_in_eval(
            self._hook_on_batch_forward_for_evaluate,  # normal forward, no ensemble distillation
            'on_batch_forward', target_hook_name='_hook_on_batch_forward')

        # prepare mixed precision computation
        self.ctx.scaler = GradScaler() if self.ctx.cfg.use_amp else None

    def parse_data(self, data):
        # modified "federatedscope.core.trainers.torch_trainer.GeneralTorchTrainer"
        #  to make it receive other splited data like "server(for recalibrate bn)".
        """Populate "${split}_data", "${split}_loader" and "num_${
        split}_data" for different data splits
        """
        init_dict = dict()
        if isinstance(data, dict):
            for split in data.keys():
                # if split not in ['train', 'val', 'test']:
                #     continue
                init_dict["{}_data".format(split)] = None
                init_dict["{}_loader".format(split)] = None
                init_dict["num_{}_data".format(split)] = 0
                if data.get(split, None) is not None:
                    if isinstance(data.get(split), Dataset):
                        init_dict["{}_data".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split))
                    elif isinstance(data.get(split), DataLoader):
                        init_dict["{}_loader".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split).dataset)
                    elif isinstance(data.get(split), dict):
                        init_dict["{}_data".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split)['y'])
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(split))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

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

        if isinstance(ctx.model, AttentiveNasDynamicModel):
            ctx.model.sample_min_subnet()
            ctx.startpoint_model = CtxVar(ctx.model.get_active_subnet(preserve_weight=True), LIFECYCLE.ROUTINE)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:

            if ctx.cur_mode == MODE.TRAIN:
                round_after = ctx.cfg.train.round_after if hasattr(ctx.cfg.train, 'round_after') else 0  # only for supernet

                ctx.cfg['train'].scheduler['multiplier'] = getattr(ctx, 'num_total_train_batch') \
                    if ctx.cfg.train.batch_or_epoch == 'batch' else getattr(ctx, 'num_train_epoch')
                ctx.cfg['train'].scheduler['max_iters'] = ctx.cfg.federate.total_round_num - round_after
                ctx.cfg['train'].scheduler['last_epoch'] = self.get_cur_state() - round_after

            else:  # ctx.cur_mode == MODE.FINETUNE
                ctx.cfg['finetune'].scheduler['multiplier'] = getattr(ctx, 'num_total_finetune_batch') \
                    if ctx.cfg.finetune.batch_or_epoch == 'batch' else getattr(ctx, 'num_finetune_epoch')
                ctx.cfg['finetune'].scheduler['max_iters'] = 1  # finetune 1 epoch, but still have "local_update_steps"
                ctx.cfg['finetune'].scheduler['last_epoch'] = -1

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
        labels = None
        if ctx.cfg.public_data.annotated: # public labels available
            _, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        ensemble_y_logits = None
        if ctx.cfg.ensemble_distillation.enable:  # ensemble distillation
            ensemble_y_logits = calculate_ensemble_logits(x, self.ensemble_models, ctx.cfg.use_amp,
                                                          ctx.cfg.ensemble_distillation.type)

        with autocast(enabled=ctx.cfg.use_amp):
            prob = ctx.model(x)
            loss_batch = calculate_mixed_loss(ctx, prob, labels, ensemble_y_logits)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

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

        num_arch_training = ctx.cfg.supernet_arch_sampler.num_arch_training
        prob = {arch_id: None for arch_id in range(num_arch_training)}

        loss_batch = []

        inplace_y_logits = None
        if ctx.cfg.inplace_distillation.enable:  # disable ensemble distillation
            ctx.model.sample_min_subnet() if ctx.cfg.inplace_distillation.type == "reverse" else ctx.model.sample_max_subnet()
            ctx.model.set_dropout_rate(0, 0, True)  # NOTE(Variant): derive logits, so disable dropout etc.

            ctx_mgr = torch.no_grad() if labels is None and ensemble_y_logits is None else contextlib.nullcontext()
            with ctx_mgr:  # 若labels或ensemble_y_logits均为none时，此时inplace distillation中teacher subnet无法获得训练目标，因此torch.no_grad
                with autocast(enabled=ctx.cfg.use_amp):  # TODO(Variant): 原需设定ctx.model.eval()，而supernet不支持eval
                    prob[0] = ctx.model(x)

            loss_batch.append(calculate_mixed_loss(ctx, prob[0], labels, ensemble_y_logits))

            with torch.no_grad():
                inplace_y_logits = prob[0].clone().detach()

        # TODO(Variant): 若使能 inplace_distillation，则 true labels 与 ensemble logits 只可被 teacher_subnet 获取(access)，
        #  获取该 teacher subnet 的 inplace_logits 用于监督其他 subnet 的训练。
        # TODO(Variant): 若不使能 inplace_distillation，则 true labels 与 ensemble logits 用于监督所有 subnet 的训练。

        # TODO(Variant): 2023.9.7 上述信息需改动，inplace_distillation存在时，label也需监督所有subnet的训练。

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
                # if ctx.cfg.inplace_distillation.enable:
                #     loss = ctx.criterion(prob[arch_id], inplace_y_logits)
                # else:
                #     loss = calculate_mixed_loss(ctx, prob[arch_id], labels, ensemble_y_logits)
                # TODO(Variant): 2023.9.7 使能这里 #question
                if ctx.cfg.inplace_distillation.enable:
                    loss = ctx.criterion(prob[arch_id], inplace_y_logits / ctx.cfg.inplace_distillation.temperature) + \
                           calculate_mixed_loss(ctx, prob[arch_id], labels, ensemble_y_logits)
                loss_batch.append(loss)

        # restore default setting
        ctx.model.sample_max_subnet()
        ctx.model.set_dropout_rate(0, 0, True)

        ctx.y_true = CtxVar(labels if labels is not None else [_.to(ctx.device) for _ in ctx.data_batch][1], LIFECYCLE.BATCH)
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

        if isinstance(ctx.model, AttentiveNasDynamicModel) and \
                ctx.cfg.freeze_teacher_subnet.enable and \
                self.get_cur_state() >= ctx.cfg.freeze_teacher_subnet.round_after:
            ctx.model.load_weights_from_pretrained_submodel(ctx.startpoint_model)

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


def calculate_mixed_loss(ctx, pred, labels, ensemble_y_logits):
    labels_loss = torch.nn.functional.cross_entropy(  # TODO(Variant): cross_entropy 无需 balanced loss
        pred, labels, label_smoothing=0 if ctx.cfg.data.type == "cifar10" else 0.1
    ) if labels is not None else 0
    ensemble_distillation_loss = ctx.criterion(pred, ensemble_y_logits / ctx.cfg.ensemble_distillation.temperature) \
        if ensemble_y_logits is not None else 0
    return labels_loss + ensemble_distillation_loss


# def call_nas_trainer(trainer_type):
#     if trainer_type == 'ensemble_distillation_trainer':
#         trainer_builder = EnsembleDistillationTrainer
#         return trainer_builder
#
#
# register_trainer('ensemble_distillation_trainer', call_nas_trainer)
