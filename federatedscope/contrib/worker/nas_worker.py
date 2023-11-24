import logging
import copy
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from functools import partial

from federatedscope.register import register_worker
from federatedscope.contrib.worker.enhance_worker import EnhanceServer, EnhanceClient

from federatedscope.core.auxiliaries.utils import merge_param_dict

from federatedscope.core.message import Message

from federatedscope.contrib.trainer.ensemble_distill_supernet_trainer import EnsembleDistillSupernetTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your worker here.
class NASServer(EnhanceServer):  # actually, it should inherit from KEMFServer
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):

        super(NASServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy,
                                        unseen_clients_id, **kwargs)

        # bind to EnsembleDistillSupernetTrainer
        assert isinstance(self.trainer, EnsembleDistillSupernetTrainer)

        # construct server_model
        self.models[0].sample_min_subnet()
        self.server_model = self.models[0].get_active_subnet(preserve_weight=True)

        self.eval_supernet = partial(eval_supernet, self)

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer. When enough messages are receiving, \
        some events (such as perform aggregation, evaluation, and move to \
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for \
                evaluation; and check the message buffer for training \
                otherwise.
            min_received_num: number of minimal received message, used for \
                async mode
        """
        if min_received_num is None:
            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()

                # NOTE(Variant): ---------------------------------------------------------------------------------------
                # before distill_trainer, test the aggregated parameter
                self.eval_supernet(DISPLAY="Supernet(agg)", spec_subnet="min", recalibrate_bn=False)

                _, _, results = self.trainer.train()  # return: num_samples, model_para, results_raw
                # save train results (train results is calculated based on ensemble models' soft logit)
                train_log_res = self._monitor.format_eval_res(
                    results,
                    rnd=self.state,
                    role='Supernet(teacher) #{}'.format(self.ID),
                    return_raw=True)
                logger.info(train_log_res)
                if self._cfg.wandb.use and self._cfg.wandb.server_train_info:
                    self._monitor.save_formatted_results(train_log_res,
                                                         save_file_name="")
                # save server model
                torch.save({'round': self.state, 'model': self.model.state_dict()},
                           os.path.join(self._cfg.outdir, 'checkpoints', f"supernet.pth"))
                # NOTE(Variant): ---------------------------------------------------------------------------------------

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval_supernet(spec_subnet="max", recalibrate_bn=True)
                    self.eval_supernet(spec_subnet="random", recalibrate_bn=True)
                    self.eval_supernet(spec_subnet="min", recalibrate_bn=True)
                    # Check and save
                    self.check_and_save()  # self.broadcast_model_para -> client -> callback_funcs_for_metrics中包含check_and_save()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()

                    # NOTE(Variant): Clean cached client models
                    for model_idx in range(self.model_num):
                        self.trainers[model_idx].ensemble_models.clear()  # very important!

                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval_supernet(spec_subnet="max", recalibrate_bn=True)
                    self.eval_supernet(spec_subnet="random", recalibrate_bn=True)
                    self.eval_supernet(spec_subnet="min", recalibrate_bn=True)
                    # Check and save
                    self.check_and_save()  # self.broadcast_model_para -> client -> callback_funcs_for_metrics中包含check_and_save()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            staleness = list()

            for client_id in train_msg_buffer.keys():
                msg_list.append(train_msg_buffer[client_id])

                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                msg_list.append(content)

                staleness.append((client_id, self.state - state))

            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # NOTE(Variant): before aggregation, cache all client models for distillation
            training_set_size = 0
            for i in range(len(msg_list)):
                sample_size, _ = msg_list[i]
                training_set_size += sample_size

            for i in range(len(msg_list)):  # recover and store the received clients' models
                local_sample_size, local_model_para = msg_list[i]
                weight = local_sample_size / training_set_size

                local_model = copy.deepcopy(self.server_model)
                # model_state_dict = local_model.state_dict()
                # model_state_dict.update(local_model_para)
                local_model.load_state_dict(local_model_para, strict=True)  # recover client model
                local_model.to(self.device)

                self.trainers[model_idx].ensemble_models.append(
                    (weight, local_model)
                )

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
            }
            # logger.info(f'The staleness is {staleness}')
            # if training_set_size > 0:
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(self.server_model.state_dict().copy(), result)
            self.server_model.load_state_dict(merged_param, strict=True)
            # model.load_weights_from_pretrained_submodel(self.server_model.state_dict())  # Deleted
        return aggregated_num

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        # NOTE(Variant): broadcast attentive_min_subnet state_dict()
        # self.models[0].sample_min_subnet()   # Deleted
        # min_subnet = self.models[0].get_active_subnet(preserve_weight=True)   # Deleted
        # model_para = min_subnet.state_dict()   # Deleted
        model_para = self.server_model.state_dict()

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


def eval_supernet(self, DISPLAY="Supernet", spec_subnet="random", recalibrate_bn=False):

    if spec_subnet == "min":
        self.model.sample_min_subnet()
    elif spec_subnet == "max":
        self.model.sample_max_subnet()
    else:
        self.model.sample_active_subnet()

    subnet = self.model.get_active_subnet(preserve_weight=True)
    subnet.to(self.device)

    metrics = {}
    if recalibrate_bn:
        torch.optim.swa_utils.update_bn(self.data["server"], subnet, device=self.device)

    for split in self._cfg.eval.split:
        results = fast_eval(subnet, self.data[split], device=self.device, use_amp=self._cfg.use_amp, header=split)
        metrics.update(results)

    formatted_eval_res = self._monitor.format_eval_res(
        metrics,
        rnd=self.state,
        role=f'{DISPLAY}({spec_subnet}) #',
        forms=self._cfg.eval.report,
        return_raw=True)

    self._monitor.save_formatted_results(formatted_eval_res)
    logger.info(formatted_eval_res)


@torch.no_grad()
def fast_eval(model, loader, device, use_amp=True, header="NULL"):
    model.eval()

    y_true, y_pred = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast(enabled=use_amp):
            y_logits = model(inputs)

        y_true.append(labels.cpu().numpy())
        y_pred.append(np.argmax(y_logits.cpu().numpy(), axis=-1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return {f'{header}_acc': np.sum(y_true == y_pred) / len(y_true), f'{header}_total': len(y_true)}


def call_nas_fl_worker(method):
    if method == "NAS" or method == 'nas':
        worker_builder = {'client': EnhanceClient, 'server': NASServer}
        return worker_builder


register_worker('NAS', call_nas_fl_worker)
