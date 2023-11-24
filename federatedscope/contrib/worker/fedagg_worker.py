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
class FedAggServer(EnhanceServer):
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

        super(FedAggServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy,
                                           unseen_clients_id, **kwargs)

        self.client_models = list()
        self.consensus_dataset = None

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        if min_received_num is None:
            min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result and self._cfg.federate.mode.lower() == "standalone":
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
                for client_id in range(1, min_received_num+1):  # cache client models
                    msg_list.append(train_msg_buffer[client_id])
                    training_set_size += train_msg_buffer[client_id][0]  # num_samples

                for num_sample, local_model in msg_list:
                    self.client_models.append(
                        (num_sample / training_set_size, local_model.to(self.device))
                    )

                self.model = hetero_aggregate(self.model, self.client_models)

                for _, local_model in self.client_models:
                    self.model.set_active_subnet(**local_model.subnet_settings)
                    state_dict = self.model.get_active_subnet(preserve_weight=True).state_dict()
                    local_model.load_state_dict(state_dict)
                    # TODO(Variant): is it necessary?
                    torch.optim.swa_utils.update_bn(self.data['server'], local_model, device=self.device)

                # NOTE(Variant): ---------------------------------------------------------------------------------------

                self.state += 1

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round #{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()

                    # Start a new training round
                    self.broadcast_model_para(msg_type='model_para',
                                              sample_client_num=self.sample_client_num)  # TODO(Variant): 改写broadcast model para
                    self.client_models.clear()
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    # self.no_broadcast_evaluation_in_clients(msg_type='no_broadcast_evaluate_after_ft')  # Preform finetune evaluation in clients
                    self.broadcast_evaluation_in_clients(
                        msg_type='evaluate_after_ft')  # Preform finetune evaluation in clients

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

    # 修改代码
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

        for cid in receiver:
            content = None if rnd == 0 else copy.deepcopy(self.client_models[cid-1][1].state_dict())
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=cid,
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=content))  # broadcast personalized model

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class FedAggClient(EnhanceClient):
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

        super(FedAggClient, self).__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client,
                                           *args, **kwargs)

    def callback_funcs_for_model_para(self, message: Message):  # actually, it received consensus logits to avoid register new callback funcs.
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        if content is not None:
            self.trainer.update(content, strict=True)
        self.state = round

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


def model_fill(model, fill=1.0):
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.fill_(fill)
    model.load_state_dict(state_dict)
    return model


def model_scale(model, multiplier=1.0):
    # for param in model.parameters():
    #     param.data.mul_(multiplier)
    # for m in model.modules():
    #     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
    #         m.running_mean.fill_(0)
    #         m.running_var.fill_(1)

    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v * multiplier
    model.load_state_dict(state_dict)
    return model


def model_clamp(model, min=0.0, max=1.0):
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = torch.clamp(v, min, max)
    model.load_state_dict(state_dict)
    return model


def model_multiply(model, rmodel):
    state_dict = model.state_dict()
    rstate_dict = rmodel.state_dict()
    for (k, v1), (_, v2) in zip(state_dict.items(), rstate_dict.items()):
        state_dict[k] = v1 * v2
    model.load_state_dict(state_dict)
    return model


def model_divide(model, rmodel):
    state_dict = model.state_dict()
    rstate_dict = rmodel.state_dict()
    for (k, v1), (_, v2) in zip(state_dict.items(), rstate_dict.items()):
        state_dict[k] = v1 / v2
    model.load_state_dict(state_dict)
    return model


def model_sum(model, rmodel):
    # for p1, p2 in zip(model.parameters(), rmodel.parameters()):
    #     p1.data.add_(p2.data)
    # for m1, m2 in zip(model.modules(), rmodel.parameters()):
    #     if isinstance(m1, torch.nn.modules.batchnorm._BatchNorm):
    #         m1.running_mean.add_(m2.running_mean)
    #         m1.running_var.add_(m2.running_var)

    state_dict = model.state_dict()
    rstate_dict = rmodel.state_dict()
    for (k, v1), (_, v2) in zip(state_dict.items(), rstate_dict.items()):
        state_dict[k] = v1 + v2
    model.load_state_dict(state_dict)
    return model


def model_patch_zero(model, patch=1.0):
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        v[v == 0.0] = 1.0
        state_dict[k] = v
    model.load_state_dict(state_dict)
    return model


def hetero_aggregate(supernet, client_models):
    # one stores parameters, the other one stores corresponding weights(coefficient).
    supernet = model_fill(copy.deepcopy(supernet), fill=0.0)  # clip即可
    weight_supernet = model_fill(copy.deepcopy(supernet), fill=0.0)  # clip即可

    # TODO(Variant): aggregate personalized models
    for client_weight, client_model in client_models:
        # prepare supernet on client node
        client_supernet = copy.deepcopy(supernet)
        model_scale(client_model, multiplier=client_weight)  # scale client models
        client_supernet.load_weights_from_pretrained_submodel(
            client_model, submodel_setting=client_model.subnet_settings
        )

        # prepare weight_supernet on client node
        client_weight_supernet = copy.deepcopy(weight_supernet)
        weight_model = model_fill(copy.deepcopy(client_model), fill=client_weight)
        client_weight_supernet.load_weights_from_pretrained_submodel(
            weight_model, submodel_setting=weight_model.subnet_settings
        )

        # sum to server model
        model_sum(supernet, client_supernet)
        model_sum(weight_supernet, client_weight_supernet)

    # model_clamp(weight_supernet, min=0.0, max=1.0)
    weight_supernet = model_patch_zero(weight_supernet, patch=1.0)
    supernet = model_divide(supernet, weight_supernet)
    return supernet


def call_fedagg_worker(method):
    if method == 'FedAgg' or method == 'fedagg':
        worker_builder = {'client': FedAggClient, 'server': FedAggServer}
        return worker_builder


register_worker('FedAgg', call_fedagg_worker)
