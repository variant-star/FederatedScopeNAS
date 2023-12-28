import logging
import copy
import os
import torch
import numpy as np
from torch.cuda.amp import autocast

from federatedscope.core.message import Message
from federatedscope.register import register_worker
from federatedscope.contrib.worker.enhance_worker import EnhanceServer, EnhanceClient


from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

from federatedscope.contrib.auxiliaries.ensemble_related import calculate_ensemble_logits

from federatedscope.core.auxiliaries.utils import merge_dict_of_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build Enhanced worker (bind with Enhanced trainer)
class FedMDServer(EnhanceServer):
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

        super(FedMDServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy,
                                          unseen_clients_id, **kwargs)

        self.client_models = list()
        self.consensus_dataset = None

    @torch.no_grad()
    def get_consensus_dataset(self):

        total_inputs, total_logits = [], []
        for inputs, _ in self.data["server"]:
            inputs = inputs.to(self.device)
            with autocast(enabled=True):
                logits = calculate_ensemble_logits(inputs, self.client_models,
                                                   logits_type=self._cfg.consensus.type)
            # save data
            total_inputs.append(inputs.cpu())
            total_logits.append(logits.cpu())

        total_inputs = torch.concatenate(total_inputs)
        total_logits = torch.concatenate(total_logits)

        self.consensus_dataset = torch.utils.data.TensorDataset(total_inputs, total_logits)
        return

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
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

                # NOTE(Variant): ---------------------------------------------------------------------------------------
                self.no_broadcast_evaluation_in_clients(
                    msg_type='no_broadcast_evaluate_no_ft')  # Preform evaluation in clients

                # Receiving enough feedback in the training process
                msg_list = list()

                training_set_size = 0.0
                train_msg_buffer = self.msg_buffer['train'][self.state]
                for client_id in train_msg_buffer.keys():  # cache client models
                    msg_list.append(train_msg_buffer[client_id])
                    training_set_size += train_msg_buffer[client_id][0]  # num_samples

                for num_sample, local_model in msg_list:
                    self.client_models.append(
                        (num_sample / training_set_size, local_model.to(self.device))
                    )

                self.get_consensus_dataset()

                self.client_models.clear()

                # NOTE(Variant): ---------------------------------------------------------------------------------------

                self.state += 1

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(f'----------- Starting a new training round (Round #{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()

                    # Start a new training round
                    self.broadcast_model_para(msg_type='model_para',
                                              sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished!')
            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def callback_funcs_model_para(self, message: Message):  # actually, it received model directly to avoid register new callback funcs.
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content
        elif round >= self.state - self.staleness_toleration:
            # Save the staled messages
            self.staled_msg_buffer.append((round, sender, content))
        else:
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date message from round #{round}')
            self.dropout_num += 1

        move_on_flag = self.check_and_move_on()

        return move_on_flag

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):  # actually, it broadcasts computed consensus logits.
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

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=self.consensus_dataset))  # broadcast consensus dataset

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class FedMDClient(EnhanceClient):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):

        super(FedMDClient, self).__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client,
                                          *args, **kwargs)

    def callback_funcs_for_model_para(self, message: Message):  # actually, it received consensus logits to avoid register new callback funcs.
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        self.state = round

        _, _, results = self.trainer.aux_train(content)  # TODO(Variant): 尚未实现

        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}(aux_train)'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        sample_size, _, results = self.trainer.train()

        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        # Return the feedbacks to the server after local update
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, copy.deepcopy(self.trainer.model))))  # communicate model directly


def call_fedmd_worker(method):
    if method == 'FedMD' or method == 'fedmd':
        worker_builder = {'client': FedMDClient, 'server': FedMDServer}
        return worker_builder


register_worker('FedMD', call_fedmd_worker)
