import torch
from torch.cuda.amp import GradScaler, autocast

from federatedscope.register import register_trainer
from federatedscope.contrib.trainer.enhance_trainer import EnhanceTrainer

from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler


# Build your trainer here.
class FedMDTrainer(EnhanceTrainer):
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

        super(FedMDTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)

        self.aux_criterion = get_criterion("kl_divergence", self.ctx.device)

    def aux_train(self, dataset):
        if dataset is None:  # skip the first broadcast
            return None, None, {}

        dataloader = get_dataloader(dataset, self.ctx.cfg, split="server")

        self.ctx.cfg['train'].scheduler['multiplier'] = 1
        self.ctx.cfg['train'].scheduler['max_iters'] = self.ctx.cfg.federate.total_round_num
        self.ctx.cfg['train'].scheduler['last_epoch'] = self.get_cur_state()

        # Initialize optimizer here to avoid the reuse of optimizers
        optimizer = get_optimizer(self.model, **self.ctx.cfg["train"].optimizer)
        scheduler = get_scheduler(optimizer, **self.ctx.cfg["train"].scheduler)  # necessary, build scheduler to change optimizer lr

        scaler = GradScaler() if self.ctx.cfg.use_amp else None

        self.model.to(self.ctx.device)

        total_num_samples = 0
        total_loss = 0

        for inputs, logits in dataloader:
            inputs, logits = inputs.to(self.ctx.device), logits.to(self.ctx.device)
            with autocast(enabled=self.ctx.cfg.use_amp):
                outputs = self.model(inputs)
                loss = self.aux_criterion(outputs, logits)

            total_num_samples += inputs.size(0)
            total_loss += loss.item() * inputs.size(0)

            optimizer.zero_grad()

            if self.ctx.cfg.use_amp:
                scaler.scale(loss).backward()
                if self.ctx.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.ctx.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.ctx.grad_clip)
                optimizer.step()

        self.discharge_model()

        return (total_num_samples, self.model.state_dict(),
                {'server_total': total_num_samples, 'server_avg_loss': total_loss/total_num_samples, 'server_loss': total_loss})


def call_fedmd_trainer(trainer_type):
    if trainer_type == 'fedmd_trainer':
        trainer_builder = FedMDTrainer
        return trainer_builder


register_trainer('fedmd_trainer', call_fedmd_trainer)
