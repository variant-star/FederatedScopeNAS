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

from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

from federatedscope.core.auxiliaries.utils import merge_param_dict

from federatedscope.core.trainers.enums import MODE

from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict_of_results

from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your worker here.
class NASServer(EnhanceServer):
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

        self.trainer.get_cur_state = self.get_cur_state

        # Initialize cached message buffer
        self.cached_client_models = {0: list()}
        self.trainer.ensemble_models = self.cached_client_models[0]  # model_num = 1

        # construct server_model
        self.models[0].sample_min_subnet()
        self.server_model = self.models[0].get_active_subnet(preserve_weight=True)

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
                self.eval_supernet(DISPLAY="Server supernet(agg)", mode=MODE.TEST, spec_subnet=["min"])

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
                torch.save({'cur_round': self.state, 'model': self.model.state_dict()},
                           os.path.join(self._cfg.outdir, f"supernet.pth"))
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
                        self.cached_client_models[model_idx].clear()  # very important!

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
        # self.broadcast_model_para(msg_type='evaluate_ignore_received_param',
        #                           filter_unseen_clients=False)
        self.broadcast_model_para(msg_type='evaluate',
                                  filter_unseen_clients=False)

    def eval_supernet(self, DISPLAY="Server Supernet", spec_subnet=["max", "random", "min"], mode=MODE.TEST):
        # evaluate supernet model
        header_to_sample = {
            "max": self.trainer.ctx.model.sample_max_subnet,
            "random": self.trainer.ctx.model.sample_active_subnet,
            "min": self.trainer.ctx.model.sample_min_subnet
        }

        for header in spec_subnet:
            header_to_sample[header]()

            metrics = {}
            for split in self._cfg.eval.split:
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split, mode=mode)  # supernet does not support 'eval' mode.
                # supernet bn_momentum is set to 0, so the bn statistics will not be calculated.
                # # and DynamicBN only supports BN
                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role=f'{DISPLAY}({header}) #',
                forms=self._cfg.eval.report,
                return_raw=True)

            self._monitor.save_formatted_results(formatted_eval_res)
            logger.info(formatted_eval_res)

            # # save supernet model
            # torch.save({'cur_round': self.state, 'model': self.supernet.state_dict()},
            #            os.path.join(self._cfg.outdir, f"supernet.pth"))

    def eval_supernet_with_recalibrate_bn(self, DISPLAY="Server Supernet", spec_subnet=["max", "random", "min"]):
        # evaluate supernet model
        header_to_sample = {
            "max": self.trainer.ctx.model.sample_max_subnet,
            "random": self.trainer.ctx.model.sample_active_subnet,
            "min": self.trainer.ctx.model.sample_min_subnet
        }

        for header in spec_subnet:

            metrics = {}

            header_to_sample[header]()

            subnet = self.trainer.ctx.model.get_active_subnet(preserve_weight=True)
            subnet.to(self.trainer.ctx.device)

            # re-calibrate-bn
            recalibrate_bn_split = "val"
            ctx_split_loader_init(self.trainer.ctx, recalibrate_bn_split)
            recalibrate_bn(subnet,
                           bn_recalibration_loader=self.trainer.ctx.get(f"{recalibrate_bn_split}_loader"),
                           num_batch_per_epoch=getattr(self.trainer.ctx, f"num_{recalibrate_bn_split}_batch"),
                           device=self.trainer.ctx.device)

            # start eval subnet
            for split in self._cfg.eval.split:
                ctx_split_loader_init(self.trainer.ctx, split)
                eval_one_epoch(subnet,
                               data_loader=self.trainer.ctx.get(f"{split}_loader"),
                               num_batch_per_epoch=getattr(self.trainer.ctx, f"num_{split}_batch"),
                               split=split, metrics=metrics,
                               device=self.trainer.ctx.device,
                               **{"use_amp": self._cfg.use_amp})

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role=f'{DISPLAY}({header})w/cali_bn #',
                forms=self._cfg.eval.report,
                return_raw=True)

            self._monitor.save_formatted_results(formatted_eval_res)
            logger.info(formatted_eval_res)

            # # save server model
            # torch.save({'cur_round': self.state, 'model': self.supernet.state_dict()},
            #            os.path.join(self._cfg.outdir, f"supernet.pth"))


    def eval(self):
        """
        To conduct evaluation. When ``cfg.federate.make_global_eval=True``, \
        a global evaluation is conducted by the server.
        """
        self.eval_supernet(mode=MODE.TRAIN)
        self.eval_supernet_with_recalibrate_bn()

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

            for i in range(len(msg_list)):  # recover and store the received clients' models
                local_sample_size, local_model_para = msg_list[i]
                weight = local_sample_size / max(training_set_size, 1)

                local_model = copy.deepcopy(self.server_model)
                # model_state_dict = local_model.state_dict()
                # model_state_dict.update(local_model_para)
                local_model.load_state_dict(local_model_para, strict=True)  # recover client model
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
            if training_set_size > 0:
                result = aggregator.aggregate(agg_info)
                # Due to lazy load, we merge two state dict
                merged_param = merge_param_dict(self.server_model.state_dict().copy(), result)
                self.server_model.load_state_dict(merged_param, strict=False)
                model.load_weights_from_pretrained_submodel(self.server_model.state_dict())

        return aggregated_num

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num: the number of sampled clients in the broadcast \
                behavior. And ``sample_client_num = -1`` denotes to \
                broadcast to all the clients.
            filter_unseen_clients: whether filter out the unseen clients that \
                do not contribute to FL process by training on their local \
                data and uploading their local model update. The splitting is \
                useful to check participation generalization gap in [ICLR'22, \
                What Do We Mean by Generalization in Federated Learning?] \
                You may want to set it to be False when in evaluation stage
        """
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

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:  # NOTE(Variant): broadcast attentive_min_subnet state_dict()
            # model_para = {} if skip_broadcast else self.models[0].state_dict()
            self.models[0].sample_min_subnet()  # NOTE(Variant): new inserted
            min_subnet = self.models[0].get_active_subnet(preserve_weight=True)  # NOTE(Variant): new inserted
            model_para = {} if skip_broadcast else min_subnet.state_dict()  # NOTE(Variant): new inserted

        # quantization
        if msg_type == 'model_para' and not skip_broadcast and \
                self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits
            if self.model_num > 1:
                model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in model_para
                ]
            else:
                model_para = symmetric_uniform_quantization(model_para, nbits)

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

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


class NASClient(EnhanceClient):
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

        super(NASClient, self).__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client,
                                        *args, **kwargs)

    def callback_funcs_for_evaluate(self, message: Message):
        """
        NOTE(Variant): 理论上，自supernet采样的模型不具备合理的BN参数，因此在broadcast时需recalibrate-bn on server data.
        评估某个subnet在某个client上的性能有如下若干选择：
        1. received BN(recalibrate on server data) or recalibrate bn on server data, directly evaluate received model
        2. recalibrate bn on private client data, then evaluate the model
        3. fine-tune on private client data, then evaluate the model
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state

        self.trainer.update(message.content,
                            strict=self._cfg.federate.share_local_model)

        # 1. received BN(recalibrate on server data) or recalibrate bn on server data, directly evaluate received model
        ctx_split_loader_init(self.trainer.ctx, "server")
        recalibrate_bn(self.trainer.ctx.model,
                       bn_recalibration_loader=self.trainer.ctx.get("server_loader"),  # server data
                       num_batch_per_epoch=getattr(self.trainer.ctx, f"num_server_batch"),
                       device=self.trainer.ctx.device)

        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(
                target_data_split_name=split)
            metrics.update(**eval_metrics)
        formatted_eval_res = self._monitor.format_eval_res(
            metrics,
            rnd=self.state,
            role='Client(re_cali_bn_on_server) #{}'.format(self.ID),
            forms=['raw'],
            return_raw=True)
        logger.info(formatted_eval_res)  # NOTE(Variant): add this to output

        # 2. recalibrate bn on private client data, then evaluate the model
        ctx_split_loader_init(self.trainer.ctx, "train")
        recalibrate_bn(self.trainer.ctx.model,
                       bn_recalibration_loader=self.trainer.ctx.get("train_loader"),
                       num_batch_per_epoch=getattr(self.trainer.ctx, f"num_train_batch"),
                       device=self.trainer.ctx.device)

        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(
                target_data_split_name=split)
            metrics.update(**eval_metrics)
        formatted_eval_res = self._monitor.format_eval_res(
            metrics,
            rnd=self.state,
            role='Client(re_cali_bn_on_client) #{}'.format(self.ID),
            forms=['raw'],
            return_raw=True)
        logger.info(formatted_eval_res)  # NOTE(Variant): add this to output

        # 3. fine-tune on private client data, then evaluate the model
        for m in self.trainer.ctx.model.modules():  # reset everything!
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                m.reset_running_stats()

        metrics = {}
        if self._cfg.finetune.before_eval:
            self.trainer.finetune()
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(
                target_data_split_name=split)
            metrics.update(**eval_metrics)
        formatted_eval_res = self._monitor.format_eval_res(
            metrics,
            rnd=self.state,
            role='Client(finetune:{}) #{}'.format(self._cfg.finetune.before_eval, self.ID),
            forms=['raw'],
            return_raw=True)
        logger.info(formatted_eval_res)  # NOTE(Variant): add this to output
        self._monitor.update_best_result(self.best_results,
                                         formatted_eval_res['Results_raw'],
                                         results_type=f"client #{self.ID}")
        self.history_results = merge_dict_of_results(
            self.history_results, formatted_eval_res['Results_raw'])


def ctx_split_loader_init(ctx, cur_split):
    """
    NOTE: modified based on federatedscope.core.trainers.torch_trainer.GeneralTorchTrainer._hook_on_epoch_start
    """
    # prepare dataloader
    if ctx.get("{}_loader".format(cur_split)) is None:
        loader = get_dataloader(
            WrapDataset(ctx.get("{}_data".format(cur_split))),
            ctx.cfg, cur_split)
        setattr(ctx, "{}_loader".format(cur_split), ReIterator(loader))
    elif not isinstance(ctx.get("{}_loader".format(cur_split)),
                        ReIterator):
        setattr(ctx, "{}_loader".format(cur_split),
                ReIterator(ctx.get("{}_loader".format(cur_split))))
    else:
        ctx.get("{}_loader".format(cur_split)).reset()


def recalibrate_bn(model, bn_recalibration_loader, num_batch_per_epoch, num_epoch=1, device=torch.device("cuda:0")):
    # re-calibrate-bn
    model.to(device)
    model.eval()
    model.reset_running_stats_for_calibration()
    with torch.no_grad():
        # NOTE(Variant): 当设置bn.momentum=None时，使用cumulative moving average (simple average)，因此epoch数不影响结果
        for _ in range(num_epoch):  # re_cali_bn N epoch

            bn_recalibration_loader.reset()

            for batch_i in range(num_batch_per_epoch):
                inputs, _ = next(bn_recalibration_loader)
                inputs = inputs.to(device)
                model(inputs)
    return


def eval_one_epoch(model, data_loader, num_batch_per_epoch, split='val', metrics={}, device=torch.device("cuda:0"),
                   forward_func=None, **kwargs):
    # start eval single model single epoch
    data_loader.reset()

    y_true, y_pred = [], []
    with torch.no_grad():
        # try:
        model.eval()  # freeze again all running statistics if bn_recalibration=True.
        # except:
        #     pass

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
    if method == "NAS" or method == 'nas':
        worker_builder = {'client': NASClient, 'server': NASServer}
        return worker_builder


register_worker('NAS', call_nas_fl_worker)
