import logging
import copy
import torch
import torch.nn as nn
import os

from federatedscope.register import register_worker
from federatedscope.contrib.worker.enhance_worker import EnhanceServer, EnhanceClient

from federatedscope.core.auxiliaries.utils import merge_param_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your worker here.
class TrainableServer(EnhanceServer):
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

        super(TrainableServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy,
                                              unseen_clients_id, **kwargs)

        # # The `self.trainer` has been build, change `make_global_eval` to false!
        # self._cfg.defrost()
        # self._cfg.federate.make_global_eval = False
        # self._cfg.freeze(inform=False, save=False, check_cfg=False)

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
                self.broadcast_evaluation_in_clients()  # Preform evaluation in clients

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

                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.broadcast_finetune_evaluation_in_clients()  # Preform finetune evaluation in clients
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag


def call_trainable_worker(method):
    if method == 'FedTrain' or method == 'fedtrain':
        worker_builder = {'client': EnhanceClient, 'server': TrainableServer}
        return worker_builder


register_worker('FedTrain', call_trainable_worker)