import logging
import copy
import torch
import torch.nn as nn
import os

from federatedscope.register import register_worker
from federatedscope.contrib.worker.enhance_worker import EnhanceServer, EnhanceClient

from federatedscope.core.auxiliaries.utils import merge_param_dict

from federatedscope.contrib.trainer.ensemble_distill_trainer import EnsembleDistillTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your worker here.
class KEMFServer(EnhanceServer):
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

        super(KEMFServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy,
                                         unseen_clients_id, **kwargs)

        # bind to EnsembleDistillTrainer
        assert isinstance(self.trainer, EnsembleDistillTrainer)

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
                self.no_broadcast_evaluation_in_clients(msg_type='no_broadcast_evaluate_no_ft')  # Preform evaluation in clients

                _, _, results = self.trainer.train()  # return: num_samples, model_para, results_raw
                # save train results (train results is calculated based on ensemble models' soft logit)
                train_log_res = self._monitor.format_eval_res(
                    results,
                    rnd=self.state,
                    role='Server #{}'.format(self.ID),
                    return_raw=True)
                logger.info(train_log_res)
                if self._cfg.wandb.use and self._cfg.wandb.server_train_info:
                    self._monitor.save_formatted_results(train_log_res,
                                                         save_file_name="")
                # save server model
                torch.save({'round': self.state, 'model': self.model.state_dict()},
                           os.path.join(self._cfg.outdir, 'checkpoints', f"server_model.pth"))
                # NOTE(Variant): ---------------------------------------------------------------------------------------

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

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
                        self.trainers[model_idx].ensemble_models.clear()

                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    # self.no_broadcast_evaluation_in_clients(msg_type='no_broadcast_evaluate_after_ft')  # Preform finetune evaluation in clients
                    self.broadcast_evaluation_in_clients(msg_type='evaluate_after_ft')  # Preform finetune evaluation in clients
                    self.eval()

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
                if self.model_num == 1:
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

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
                weight = local_sample_size / max(training_set_size, 1)

                local_model = copy.deepcopy(model)
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
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)

        return aggregated_num


def call_kemf_worker(method):
    if method == 'FedKEMF' or method == 'fedkemf':
        worker_builder = {'client': EnhanceClient, 'server': KEMFServer}
        return worker_builder


register_worker('FedKEMF', call_kemf_worker)