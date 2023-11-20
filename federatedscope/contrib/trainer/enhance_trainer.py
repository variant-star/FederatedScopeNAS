import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.trainers.utils import format_log_hooks

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar, lifecycle

from federatedscope.core.auxiliaries.decorators import use_diff

logger = logging.getLogger(__name__)


# Build your trainer here.
class EnhanceTrainer(GeneralTorchTrainer):
    """
        1. add torch.cuda.amp,  2. enhance optimizer and lr_scheduler
        WARNING: the easy and best practise is just like federatedscope/contrib/trainer/torch_example.py.
                 The context manager is not necessary.
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None
                 ):

        self.model = model

        super(EnhanceTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        if only_for_eval:
            # only_for_eval => don't register hooks_train. So we change it.
            # However, this change will not display in self.print_trainer_meta_info()
            self.register_default_hooks_train()

        self.replace_hook_in_train(
            self._hook_on_batch_forward_for_train,
            'on_batch_forward', target_hook_name='_hook_on_batch_forward')

        self.replace_hook_in_eval(
            self._hook_on_fit_start_init_for_evaluate,
            'on_fit_start', target_hook_name='_hook_on_fit_start_init')
        # Important! for supernet, it doesn't support 'eval' mode, so we use "train" mode, but no optimizer and scheduler

        self.replace_hook_in_eval(
            self._hook_on_batch_forward_for_evaluate,  # normal forward, no optimizer and lr_scheduler.
            'on_batch_forward', target_hook_name='_hook_on_batch_forward')

        if self._cfg.finetune.before_eval:
            self.replace_hook_in_ft(
                self._hook_on_batch_forward_for_finetune,
                'on_batch_forward', target_hook_name='_hook_on_batch_forward')

        # prepare mixed precision computation
        self.ctx.scaler = GradScaler() if self.ctx.cfg.use_amp else None

    def reset_hook_in_ft(self, target_trigger, target_hook_name=None):
        """
        modified from reset_hook_in_train. It's so weird not having this method.
        """
        hooks_dict = self.hooks_in_ft
        del_one_hook_idx = self._reset_hook_in_trigger(hooks_dict,
                                                       target_hook_name,
                                                       target_trigger)
        return del_one_hook_idx

    def replace_hook_in_ft(self, new_hook, target_trigger,
                           target_hook_name):
        """
        modified from replace_hook_in_train. It's so weird not having this method.
        """
        del_one_hook_idx = self.reset_hook_in_ft(
            target_trigger=target_trigger, target_hook_name=target_hook_name)
        self.register_hook_in_ft(new_hook=new_hook,
                                 trigger=target_trigger,
                                 insert_pos=del_one_hook_idx)

    @use_diff
    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        if self.cfg.train.local_update_steps > 0 and self.ctx.check_split(target_data_split_name, skip=True):
            num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                            target_data_split_name)
        else:
            num_samples = 1  # set min samples = 1
            self.ctx.eval_metrics = dict()
        return num_samples, self.get_model_para(), self.ctx.eval_metrics

    def evaluate(self, target_data_split_name="test", hooks_set=None, mode=MODE.TEST):
        hooks_set = hooks_set or self.hooks_in_eval

        if self.ctx.check_split(target_data_split_name, skip=True):
            with torch.no_grad():
                self._run_routine(mode, hooks_set, target_data_split_name)
        else:
            self.ctx.eval_metrics = dict()

        return self.ctx.eval_metrics

    def finetune(self, target_data_split_name="train", hooks_set=None):
        """
        modified from federatedscope.core.trainers.trainer.Trainer.finetune
        """
        hooks_set = hooks_set or self.hooks_in_ft

        if self.cfg.finetune.local_update_steps > 0 and self.ctx.check_split(target_data_split_name, skip=True):
            num_samples = self._run_routine(MODE.FINETUNE, hooks_set,
                                            target_data_split_name)
        else:
            num_samples = 1  # set min samples = 1
            self.ctx.eval_metrics = dict()

        return num_samples, self.get_model_para(), self.ctx.eval_metrics

    @lifecycle(LIFECYCLE.EPOCH)
    def _run_epoch(self, hooks_set):
        """
        modified from federatedscope.core.trainers.trainer.Trainer._run_epoch
        """
        for epoch_i in range(
                getattr(self.ctx, f"num_{self.ctx.cur_mode}_epoch")): # change self.ctx.cur_split into self.ctx.cur_mode
            self.ctx.cur_epoch_i = CtxVar(epoch_i, "epoch")

            for hook in hooks_set["on_epoch_start"]:
                hook(self.ctx)

            self._run_batch(hooks_set)

            for hook in hooks_set["on_epoch_end"]:
                hook(self.ctx)

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set):
        """
        modified from federatedscope.core.trainers.trainer.Trainer._run_batch
        """
        for batch_i in range(
                getattr(self.ctx, f"num_{self.ctx.cur_split}_batch")): # change self.ctx.cur_split into self.ctx.cur_mode
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)

            for hook in hooks_set["on_batch_start"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_forward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_backward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_end"]:
                hook(self.ctx)

            # Break in the final epoch
            if self.ctx.cur_mode in [
                    MODE.TRAIN, MODE.FINETUNE
            ] and self.ctx.cur_epoch_i == getattr(self.ctx, f"num_{self.ctx.cur_mode}_epoch") - 1:  # make changes
                if batch_i >= getattr(self.ctx, f"num_{self.ctx.cur_split}_batch_last_epoch") - 1:  # make changes
                    break

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
        # prepare model and optimizer
        ctx.model.to(ctx.device)

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

    def _hook_on_batch_forward_for_evaluate(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]  # NOTE(Variant): we use the true label (only for eval)
        with autocast(enabled=ctx.cfg.use_amp):
            prob = ctx.model(x)
            loss_batch = F.cross_entropy(prob, label)
        # if len(label.size()) == 0:
        #     label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(prob, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_batch, LIFECYCLE.BATCH)  # use the simplest cross_entropy
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    def _hook_on_batch_forward_for_finetune(self, ctx):
        if ctx.cfg.finetune.criterion_base == "train":
            return self._hook_on_batch_forward_for_train(ctx)
        else:
            return self._hook_on_batch_forward_for_evaluate(ctx)

    def _hook_on_batch_backward(self, ctx):
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

        if ctx.cfg[ctx.cur_mode].batch_or_epoch == 'batch':
            ctx.scheduler.step()
        else:  # batch_or_epoch == 'epoch'
            if ctx.cur_batch_i == getattr(ctx, f"num_{self.ctx.cur_split}_batch") - 1:
                ctx.scheduler.step()

    def _hook_on_fit_end(self, ctx):
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
        # 模仿分类任务评估top1 acc或correct等
        if ctx.ys_true.ndim == 2:
            ctx.ys_true = np.argmax(ctx.ys_true, axis=1)  # NOTE(Variant): new added

        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)

    def _hook_on_fit_start_init_for_evaluate(self, ctx):
        ctx.model.to(ctx.device)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_data_parallel_init(self, ctx):
        pass

    def _hook_on_batch_forward_flop_count(self, ctx):
        pass

    def _hook_on_fit_start_calculate_model_size(self, ctx):
        pass

    def print_trainer_meta_info(self):
        logger.info(f"Model meta-info: {type(self.ctx.model)}.")
        logger.debug(f"Model meta-info: {self.ctx.model}.")
        # logger.info(f"Data meta-info: {self.ctx['data']}.")

        logger.info(f"After register default hooks,\n"
                    f"\tthe hooks_in_train is:\n\t"
                    f"{format_log_hooks(self.hooks_in_train)};\n"
                    f"\tthe hooks_in_eval is:\n\
            t{format_log_hooks(self.hooks_in_eval)}")


def call_enhance_trainer(trainer_type):
    if trainer_type == 'enhance_trainer':
        trainer_builder = EnhanceTrainer
        return trainer_builder


register_trainer('enhance_trainer', call_enhance_trainer)
