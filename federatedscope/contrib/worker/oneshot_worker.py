import logging
import copy
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from federatedscope.register import register_worker
from federatedscope.contrib.worker.enhance_worker import EnhanceServer, EnhanceClient

from federatedscope.core.auxiliaries.utils import merge_param_dict

from federatedscope.core.trainers.enums import MODE

OVERWRITE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your worker here.
class OneShotServer(EnhanceServer):
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

        super(OneShotServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy,
                                            unseen_clients_id, **kwargs)

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

    def eval_supernet(self, DISPLAY="Supernet", spec_subnet="random", recalibrate_bn=False, mode=None):

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

        run_mode = mode or (MODE.TEST if recalibrate_bn else MODE.TRAIN)  # 若指定mode，即测试supernet(agg)

        for split in self._cfg.eval.split:
            results = fast_eval(subnet, self.data[split], device=self.device, use_amp=self._cfg.use_amp,
                                mode=run_mode, header=split)
            metrics.update(results)

        formatted_eval_res = self._monitor.format_eval_res(
            metrics,
            rnd=self.state,
            role=f'{DISPLAY}({spec_subnet}){"(MODE.TEST)" if run_mode == MODE.TEST else "(MODE.TRAIN)"} #',
            forms=self._cfg.eval.report,
            return_raw=True)

        self._monitor.save_formatted_results(formatted_eval_res)
        logger.info(formatted_eval_res)


def fast_eval(model, loader, device, use_amp=True, mode=MODE.TEST, header="NULL"):
    y_true, y_pred = [], []
    with torch.no_grad():
        model.eval() if mode == MODE.TEST else model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(enabled=use_amp):
                y_logits = model(inputs)

            y_true.append(labels.cpu().numpy())
            y_pred.append(np.argmax(y_logits.cpu().numpy(), axis=-1))

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

    return {f'{header}_acc': np.sum(y_true == y_pred) / len(y_true), f'{header}_total': len(y_true)}


def call_oneshot_fl_worker(method):
    if method == "OneShot" or method == 'oneshot':
        worker_builder = {'client': EnhanceClient, 'server': OneShotServer}
        return worker_builder


register_worker('OneShot', call_oneshot_fl_worker)
