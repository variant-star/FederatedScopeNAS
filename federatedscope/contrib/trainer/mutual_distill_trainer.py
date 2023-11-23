import os
import copy
import torch
from torch.cuda.amp import GradScaler, autocast

from federatedscope.register import register_trainer
from federatedscope.contrib.trainer.enhance_trainer import EnhanceTrainer

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar, lifecycle

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

from federatedscope.core.auxiliaries.model_builder import get_model, get_trainable_para_names
from federatedscope.core.trainers.utils import format_log_hooks, \
    filter_by_specified_keywords

from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict


# Build your trainer here.
class MutualDistillTrainer(EnhanceTrainer):
    """
        distillate knowledge from 'ensemble client models' to 'nas supernet'
        - note: only can be used in Client. (not sure the use in Server.)
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None
                 ):

        super(MutualDistillTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        aux_model_cfg = self._cfg.model.clone()
        aux_model_cfg.defrost()
        aux_model_cfg.type = "attentive_min_subnet"
        self.aux_model = get_model(aux_model_cfg, local_data=None, backend=self._cfg.backend)

    def _hook_on_fit_start_init(self, ctx):
        # prepare model and optimizer
        ctx.model.to(ctx.device)
        self.aux_model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:

            if ctx.cur_mode == MODE.TRAIN:
                ctx.cfg['train'].scheduler['multiplier'] = getattr(ctx, 'num_total_train_batch') \
                    if ctx.cfg.train.batch_or_epoch == 'batch' else getattr(ctx, 'num_train_epoch')
                ctx.cfg['train'].scheduler['max_iters'] = ctx.cfg.federate.total_round_num
                ctx.cfg['train'].scheduler['last_epoch'] = self.get_cur_state()

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

            if ctx.cur_mode in [MODE.FINETUNE] and ctx.cfg.finetune.criterion_base == "train":
                ctx.aux_optimizer = get_optimizer(self.aux_model,
                                                  **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.aux_scheduler = get_scheduler(ctx.aux_optimizer,
                                                  **ctx.cfg[ctx.cur_mode].scheduler)

        # TODO: the number of batch and epoch is decided by the current mode
        #  and data split, so the number of batch and epoch should be
        #  initialized at the beginning of the routine

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward_for_train(self, ctx):
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        # mutual learning   # TODO(Variant): 这个位置的.train .train好像不必要，在context中以change mode实现，然而这里涉及到另一个模型
        ctx.model.train()
        self.aux_model.train()

        with autocast(enabled=ctx.cfg.use_amp):
            prob = ctx.model(x)
            aux_prob = self.aux_model(x)
            loss_batch = ctx.criterion(prob, aux_prob.clone().detach(), labels) +\
                         ctx.criterion(aux_prob, prob.clone().detach(), labels)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(prob), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        if hasattr(ctx, "aux_optimizer"):
            ctx.aux_optimizer.zero_grad()  # add "local aux_model"

        if ctx.cfg.use_amp:
            ctx.scaler.scale(ctx.loss_task).backward()
            if ctx.grad_clip > 0:
                ctx.scaler.unscale_(ctx.optimizer)
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
                if hasattr(ctx, "aux_optimizer"):
                    ctx.scaler.unscale_(ctx.aux_optimizer)  # add "local aux_model"
                    torch.nn.utils.clip_grad_norm_(self.aux_model.parameters(),  # add "local aux_model"
                                                   ctx.grad_clip)
            ctx.scaler.step(ctx.optimizer)
            if hasattr(ctx, "aux_optimizer"):
                ctx.scaler.step(ctx.aux_optimizer)  # add "local aux_model"
            ctx.scaler.update()
        else:
            ctx.loss_task.backward()
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
                if hasattr(ctx, "aux_optimizer"):
                    torch.nn.utils.clip_grad_norm_(self.aux_model.parameters(),  # add "local aux_model"
                                                   ctx.grad_clip)
            ctx.optimizer.step()
            if hasattr(ctx, "aux_optimizer"):
                ctx.aux_optimizer.step()  # add "local aux_model"

        if ctx.cfg[ctx.cur_mode].batch_or_epoch == 'batch':
            ctx.scheduler.step()
        else:  # batch_or_epoch == 'epoch'
            if ctx.cur_batch_i == getattr(ctx, f"num_{self.ctx.cur_split}_batch") - 1:
                ctx.scheduler.step()

    def _hook_on_fit_end_for_train_mode_recalibrate_bn(self, ctx):
        torch.optim.swa_utils.update_bn(ctx.data['server'], ctx.model, ctx.device)
        torch.optim.swa_utils.update_bn(ctx.data['server'], self.aux_model, ctx.device)

    def get_model_para(self):  # change "local ctx.model" to "local aux_model"
        if self.cfg.federate.process_num > 1:
            return self._param_filter(self.aux_model.state_dict())
        else:
            return self._param_filter(
                self.aux_model.state_dict() if self.cfg.federate.
                share_local_model else self.aux_model.cpu().state_dict())

    def update(self, model_parameters, strict=False):  # change "local ctx.model" to "local aux_model"
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(self.aux_model.state_dict().copy(),
                                        self._param_filter(model_parameters))
        self.aux_model.load_state_dict(merged_param, strict=strict)

    def _param_filter(self, state_dict, filter_keywords=None):  # change "local ctx.model" to "local aux_model"
        if self.cfg.federate.method in ["local", "global"]:
            return {}

        if filter_keywords is None:
            filter_keywords = self.cfg.personalization.local_param

        trainable_filter = lambda p: True if \
            self.cfg.personalization.share_non_trainable_para else \
            p in get_trainable_para_names(self.aux_model)
        keyword_filter = filter_by_specified_keywords
        return dict(
            filter(
                lambda elem: trainable_filter(elem[0]) and keyword_filter(
                    elem[0], filter_keywords), state_dict.items()))

    def discharge_model(self):  # modified, add "local aux_model"
        # Avoid memory leak
        if not self.cfg.federate.share_local_model:
            if torch is None:
                pass
            else:
                self.ctx.model.to(torch.device("cpu"))
                self.aux_model.to(torch.device("cpu"))  # add "local aux_model"


def call_mutual_distill_trainer(trainer_type):
    if trainer_type == 'mutual_distill_trainer':
        trainer_builder = MutualDistillTrainer
        return trainer_builder


register_trainer('mutual_distill_trainer', call_mutual_distill_trainer)
