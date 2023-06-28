import logging
import copy
import torch
import os
import sys
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client

from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

from federatedscope.core.auxiliaries.utils import merge_param_dict

# 创建trainer、monitor等
from federatedscope.core.monitors.monitor import Monitor

from federatedscope.core.trainers.enums import MODE

from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict_of_results

from federatedscope.contrib.auxiliaries.ensemble_related import calculate_ensemble_logits

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your worker here.
class NASServer(Server):
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

        assert data is not None

        config = copy.deepcopy(config)
        config['client_version'] = False
        config['get_cur_state'] = None
        config['ensemble_models'] = None

        super(NASServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy,
                                        unseen_clients_id, **kwargs)

        # Initialize cached message buffer
        assert self.model_num == 1
        self.cached_client_models = {model_idx: list() for model_idx in range(self.model_num)}

        # 创建supernet
        self.supernet = get_model(self._cfg.supernet, self.data, backend=self._cfg.backend)
        # self.supernet.sample_max_subnet()
        # self.supernet = self.supernet.get_active_subnet(preserve_weight=True)

        # 修正data training或eval相关的参数
        self.server_trainer_specified_cfg = copy.deepcopy(self._cfg)
        self.server_trainer_specified_cfg.merge_from_other_cfg(self._cfg.server_trainer_specified, check_cfg=False)
        self.server_trainer_specified_cfg['get_cur_state'] = self.get_cur_state
        self.server_trainer_specified_cfg['ensemble_models'] = self.cached_client_models[0]

        self._monitor = Monitor(self.server_trainer_specified_cfg, monitored_object=self)

        self.trainer = get_trainer(
            model=self.models[0],
            data=copy.deepcopy(self.data),
            device=self.device,
            config=self.server_trainer_specified_cfg,
            only_for_eval=False,
            monitor=self._monitor
        )
        self.trainers = [self.trainer]

        # 修正data training或eval相关的参数
        self.supernet_trainer_specified_cfg = copy.deepcopy(self._cfg)
        self.supernet_trainer_specified_cfg.merge_from_other_cfg(self._cfg.supernet_trainer_specified, check_cfg=False)
        self.supernet_trainer_specified_cfg['get_cur_state'] = self.get_cur_state
        self.supernet_trainer_specified_cfg['ensemble_models'] = self.cached_client_models[0]

        self._supernet_monitor = Monitor(self.supernet_trainer_specified_cfg, monitored_object=self)

        self.supernet_trainer = get_trainer(
            model=self.supernet,
            data=copy.deepcopy(self.data),
            device=self.device,
            config=self.supernet_trainer_specified_cfg,
            only_for_eval=False,
            monitor=self._supernet_monitor
        )  # the trainer is only used for supernet training

        self.train_server_model_after_round = self.server_trainer_specified_cfg.train.round_after
        self.train_supernet_after_round = self.supernet_trainer_specified_cfg.train.round_after

        if self._cfg.server_model_bn_tracking:
            for m in self.model.modules():  # TODO(Variant):
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.track_running_stats = False
        if self._cfg.supernet_bn_tracking:
            for m in self.supernet.modules():  # TODO(Variant):
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.track_running_stats = False

    def get_cur_state(self):
        return self.state

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

                # self.eval_server_model(DISPLAY="Server Model(agg)(MODE.TEST)(before train)")
                # TODO(Variant): ---------------------------------------------------------------------------------------
                # execute distillation for server model
                if self.state >= self.train_server_model_after_round:
                    # train server model
                    _, _, results = self.trainer.train()  # return: num_samples, model_para, results_raw
                    # save train results (train results is calculated based on ensemble models' soft logit)
                    train_log_res = self._monitor.format_eval_res(
                        results,
                        rnd=self.state,
                        role='Server Model #{}'.format(self.ID),
                        return_raw=True)
                    logger.info(train_log_res)
                    if self._cfg.wandb.use and self._cfg.wandb.server_train_info:
                        self._monitor.save_formatted_results(train_log_res,
                                                             save_file_name="")
                # save server model
                torch.save({'cur_round': self.state, 'model': self.models[0].state_dict()},
                           os.path.join(self._cfg.outdir, f"server_model.pth"))

                # evaluate server model
                # self.eval_server_model(DISPLAY="Server Model(agg)(MODE.TRAIN)", mode=MODE.TRAIN)
                self.eval_server_model(DISPLAY="Server Model(agg)(MODE.TEST)")

                # # evaluate ensemble models
                # self.eval_ensemble_model()

                # load server model into supernet
                if self._cfg.overwrite_supernet_with_aggregation:
                    self.supernet.load_weights_from_pretrained_submodel(self.models[0])

                # execute distillation for supernet
                if self.state >= self.train_supernet_after_round:
                    # train supernet model
                    _, _, results = self.supernet_trainer.train()  # return: num_samples, model_para, results_raw
                    # save train results (train results is calculated based on ensemble models' soft logit)
                    train_log_res = self._supernet_monitor.format_eval_res(
                        results,
                        rnd=self.state,
                        role='Server Supernet(teacher) #{}'.format(self.ID),
                        return_raw=True)
                    logger.info(train_log_res)
                    if self._cfg.wandb.use and self._cfg.wandb.server_train_info:
                        self._monitor.save_formatted_results(train_log_res,
                                                             save_file_name="")
                    # save server model
                    torch.save({'cur_round': self.state, 'model': self.supernet.state_dict()},
                               os.path.join(self._cfg.outdir, f"supernet.pth"))

                # decouple the min subnet from supernet
                if self._cfg.overwrite_supernet_with_aggregation:
                    self.supernet.sample_min_subnet()
                    decoupled_state_dict = self.supernet.get_active_subnet(preserve_weight=True).state_dict()
                    self.model.load_state_dict(decoupled_state_dict)

                # TODO(Variant): ---------------------------------------------------------------------------------------

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
                        self.cached_client_models[model_idx] = list()

                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def distributed_client_eval(self):
        # send 'evaluation' cmd to clients
        self.broadcast_model_para(msg_type='evaluate_ignore_received_param',
                                  filter_unseen_clients=False)

    def eval_server_model(self, DISPLAY="Server Model", mode=MODE.TEST):
        from federatedscope.core.auxiliaries.utils import merge_dict_of_results

        # evaluate server model
        for i in range(self.model_num):

            metrics = {}
            for split in self.server_trainer_specified_cfg.eval.split:
                eval_metrics = self.trainers[i].evaluate(
                    target_data_split_name=split, mode=mode)  # TODO(Variant): important!
                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role=f'{DISPLAY} #',
                forms=self.server_trainer_specified_cfg.eval.report,
                return_raw=True)

            self._monitor.update_best_result(
                self.best_results,
                formatted_eval_res['Results_raw'],
                results_type="server_global_eval")
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res)
            self._monitor.save_formatted_results(formatted_eval_res)
            logger.info(formatted_eval_res)

            # # save server model
            # torch.save({'cur_round': self.state, 'model': self.models[i].state_dict()},
            #            os.path.join(self._cfg.outdir, f"server_model[i].pth"))

    def eval_supernet(self, DISPLAY="Server Supernet", spec_subnet=["max", "random", "min"], mode=MODE.TEST):
        # evaluate supernet model
        header_to_sample = {
            "max": self.supernet_trainer.ctx.model.sample_max_subnet,
            "random": self.supernet_trainer.ctx.model.sample_active_subnet,
            "min": self.supernet_trainer.ctx.model.sample_min_subnet
        }

        for header in spec_subnet:
            header_to_sample[header]()

            metrics = {}
            for split in self.supernet_trainer_specified_cfg.eval.split:
                eval_metrics = self.supernet_trainer.evaluate(
                    target_data_split_name=split, mode=mode)  # supernet does not support 'eval' mode.
                # supernet bn_momentum is set to 0, so the bn statistics will not be calculated.
                # # and DynamicBN only supports BN
                metrics.update(**eval_metrics)

            formatted_eval_res = self._supernet_monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role=f'{DISPLAY}({header}) #',
                forms=self.supernet_trainer_specified_cfg.eval.report,
                return_raw=True)

            self._supernet_monitor.save_formatted_results(formatted_eval_res)
            logger.info(formatted_eval_res)

            # # save supernet model
            # torch.save({'cur_round': self.state, 'model': self.supernet.state_dict()},
            #            os.path.join(self._cfg.outdir, f"supernet.pth"))

    def eval_supernet_with_recalibrate_bn(self, DISPLAY="Server Supernet", spec_subnet=["max", "random", "min"]):
        # evaluate supernet model
        header_to_sample = {
            "max": self.supernet_trainer.ctx.model.sample_max_subnet,
            "random": self.supernet_trainer.ctx.model.sample_active_subnet,
            "min": self.supernet_trainer.ctx.model.sample_min_subnet
        }

        for header in spec_subnet:

            metrics = {}

            header_to_sample[header]()

            subnet = self.supernet_trainer.ctx.model.get_active_subnet(preserve_weight=True)
            subnet.cuda()  # TODO(Variant): simply writing

            # re-calibrate-bn
            recalibrate_bn(subnet,
                           bn_recalibrate_bn_loader=self.trainer.ctx.get("train_loader"),
                           num_batch_per_epoch=getattr(self.trainer.ctx, f"num_train_batch"))

            # start eval subnet
            for split in self.supernet_trainer_specified_cfg.eval.split:
                eval_one_epoch(subnet,
                               data_loader=self.trainer.ctx.get(f"{split}_loader"),
                               num_batch_per_epoch=getattr(self.trainer.ctx, f"num_{split}_batch"),
                               split=split, metrics=metrics,
                               **{"use_amp": self._cfg.use_amp})

            formatted_eval_res = self._supernet_monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role=f'{DISPLAY}({header})w/cali_bn #',
                forms=self.supernet_trainer_specified_cfg.eval.report,
                return_raw=True)

            self._supernet_monitor.save_formatted_results(formatted_eval_res)
            logger.info(formatted_eval_res)

            # # save server model
            # torch.save({'cur_round': self.state, 'model': self.supernet.state_dict()},
            #            os.path.join(self._cfg.outdir, f"supernet.pth"))

    def eval_ensemble_model(self, DISPLAY="Ensemble Model"):
        formatted_results = {'Role': f'{DISPLAY} #', 'Round': self.state, 'Results_raw': {}}
        for split in self.server_trainer_specified_cfg.eval.split:
            # evaluate ensemble model
            eval_one_epoch(self.cached_client_models[0],
                           data_loader=self.trainer.ctx.get(f"{split}_loader"),
                           num_batch_per_epoch=getattr(self.trainer.ctx, f"num_{split}_batch"),
                           split=split,
                           metrics=formatted_results['Results_raw'],
                           forward_func=calculate_ensemble_logits,
                           **{"use_amp": self._cfg.use_amp,
                              "distillation_logits_type": self._cfg.ensemble_distillation.type})

        logger.info(formatted_results)
        with open(os.path.join(self._cfg.outdir, "eval_results.raw"), "a") as outfile:
            outfile.write(str(formatted_results) + "\n")

    def eval(self):
        """
        To conduct evaluation. When ``cfg.federate.make_global_eval=True``, \
        a global evaluation is conducted by the server.
        """
        self.distributed_client_eval()

        # self.eval_server_model()

        # self.eval_supernet(mode=MODE.TRAIN)
        self.eval_supernet_with_recalibrate_bn()  # TODO(Variant): important!

        # self.eval_ensemble_model()

        # Check and save
        self.check_and_save()  # self.broadcast_model_para -> client -> callback_funcs_for_metrics中包含check_and_save()

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

            for i in range(len(msg_list)):
                local_sample_size, local_model_para = msg_list[i]
                weight = local_sample_size / training_set_size

                local_model = copy.deepcopy(self.models[model_idx])
                # model_state_dict = local_model.state_dict()
                # model_state_dict.update(local_model_para)
                local_model.load_state_dict(local_model_para, strict=True)  # recover client model  # TODO(Variant): this is for my own project
                local_model.to(self.device)

                self.cached_client_models[model_idx].append(
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

    def check_and_save(self):
        """
        To save the results and save model after each evaluation, and check \
        whether to early stop.
        """

        if self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            # last round or early stopped
            self.save_best_results()
            if not self._cfg.federate.make_global_eval:
                self.save_client_eval_results()
            self.terminate(msg_type='finish')

        # Clean the clients evaluation msg buffer
        if not self._cfg.federate.make_global_eval:
            round = max(self.msg_buffer['eval'].keys())
            self.msg_buffer['eval'][round].clear()

        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            self.state += 1


class NASClient(Client):
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

        config = copy.deepcopy(config)
        config['client_version'] = True
        config['get_cur_state'] = self.get_cur_state

        super(NASClient, self).__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client,
                                        *args, **kwargs)

        self.register_handlers('evaluate_ignore_received_param', self.callback_funcs_for_evaluate_ignore_received_param, [None])

    def get_cur_state(self):
        return self.state

    def callback_funcs_for_evaluate_ignore_received_param(self, message: Message):
        """
        NOTE(Variant): only be invoked for local model evaluate, ignore the received param

        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        # if message.content is not None:  # NOTE(Variant): comment it for ignoring the received model parameters
        #     self.trainer.update(message.content,
        #                         strict=self._cfg.federate.share_local_model)
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
                "local", "global"
        ]:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms=['raw'],
                return_raw=True)
            logger.info(formatted_eval_res)  # NOTE(Variant): add this to output
            self._monitor.update_best_result(self.best_results,
                                             formatted_eval_res['Results_raw'],
                                             results_type=f"client #{self.ID}")
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])
            self.early_stopper.track_and_check(self.history_results[
                self._cfg.eval.best_res_update_round_wise_key])

        # self.comm_manager.send(
        #     Message(msg_type='metrics',
        #             sender=self.ID,
        #             receiver=[sender],
        #             state=self.state,
        #             timestamp=timestamp,
        #             content=metrics))


def recalibrate_bn(model, bn_recalibrate_bn_loader, num_batch_per_epoch, num_epoch=5, device=torch.device("cuda:0")):
    # re-calibrate-bn
    with torch.no_grad():
        model.eval()
        model.reset_running_stats_for_calibration()

        for _ in range(num_epoch):  # re_cali_bn 5 epoch

            bn_recalibrate_bn_loader.reset()

            for batch_i in range(num_batch_per_epoch):
                inputs, _ = next(bn_recalibrate_bn_loader)
                inputs = inputs.to(device)
                model(inputs)
    return


def eval_one_epoch(model, data_loader, num_batch_per_epoch, split='val', metrics={}, device=torch.device("cuda:0"),
                   forward_func=None, **kwargs):
    # start eval single model single epoch
    data_loader.reset()

    y_true, y_pred = [], []
    with torch.no_grad():
        try:
            model.eval()  # freeze again all running statistics if bn_recalibration=True.
        except:
            pass

        for batch_i in range(num_batch_per_epoch):
            inputs, labels = next(data_loader)
            inputs, labels = inputs.to(device), labels.to(device)
            if forward_func is not None:
                y_logits = forward_func(inputs, model,
                                        kwargs.get("use_amp", False),
                                        kwargs.get("distillation_logits_type", "avg_logits"),
                                        kwargs.get("mode", MODE.TEST))
            else:
                with autocast(enabled=kwargs.get("use_amp", False)):
                    y_logits = model(inputs)

            y_true.append(labels.cpu().numpy())
            y_pred.append(np.argmax(y_logits.cpu().numpy(), axis=-1))

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        metrics.update({f'{split}_acc': np.sum(y_true == y_pred) / len(y_true),
                        f'{split}_total': len(y_true)})

    return metrics


def call_nas_fl_worker(method):
    if method == 'nas_fl':
        worker_builder = {'client': NASClient, 'server': NASServer}
        return worker_builder


register_worker('nas_fl', call_nas_fl_worker)
