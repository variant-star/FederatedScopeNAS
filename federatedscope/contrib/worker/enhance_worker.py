import logging
import copy
import os
import torch

from federatedscope.core.message import Message
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client

from federatedscope.core.auxiliaries.model_builder import get_model

from federatedscope.core.auxiliaries.utils import merge_dict_of_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build Enhanced worker (bind with Enhanced trainer)
class EnhanceServer(Server):
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

        cfg = config.clone()

        # save server_config.yaml begin!
        from pathlib import Path
        Path(cfg.outdir + '/configs').mkdir(parents=True, exist_ok=True)
        Path(cfg.outdir + '/checkpoints').mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.outdir, 'configs', "server_config.yaml"), 'w') as outfile:
            from contextlib import redirect_stdout
            with redirect_stdout(outfile):
                tmp_cfg = copy.deepcopy(cfg)
                tmp_cfg.clear_aux_info()
                print(tmp_cfg.dump())
        # save server_config.yaml end!

        # load pretrained checkpoint!
        assert model is not None
        if cfg.model.type.startswith("attentive") and cfg.model.pretrain != "":
            supernet_cfg = cfg.model.clone()
            supernet_cfg.defrost()
            supernet_cfg.type = "attentive_supernet"
            supernet = get_model(supernet_cfg, local_data=None, backend=cfg.backend)
            # 加载supernet模型
            saved_state_dict = torch.load(supernet_cfg.pretrain, map_location=torch.device("cpu"))
            if "model" in saved_state_dict:
                saved_state_dict = saved_state_dict["model"]
            if "state_dict" in saved_state_dict:
                saved_state_dict = saved_state_dict["state_dict"]
            supernet.load_state_dict(saved_state_dict, strict=True)

            # supernet中采样模型加载权重
            if cfg.model.type == "attentive_min_subnet":
                supernet.sample_min_subnet()
                pretrained = supernet.get_active_subnet(preserve_weight=True).to(device)
                torch.optim.swa_utils.update_bn(data['server'], pretrained, device=device)  # recalibrate_bn
            elif cfg.model.type == "attentive_subnet":
                assert hasattr(cfg.model, 'arch_cfg')
                supernet.set_active_subnet(**cfg.model.arch_cfg)
                pretrained = supernet.get_active_subnet(preserve_weight=True).to(device)
                torch.optim.swa_utils.update_bn(data['server'], pretrained, device=device)  # recalibrate_bn
            else:  # "attentive_supernet"
                pretrained = supernet

            model.load_state_dict(pretrained.state_dict(), strict=True)
            print(f"Server{ID} Loaded pretrained checkpoint.")
            del supernet

        super(EnhanceServer, self).__init__(ID, state, cfg, data, model, client_num, total_round_num, device, strategy,
                                            unseen_clients_id, **kwargs)

        # # The `self.trainer` has been build, change `make_global_eval` to false!
        # self._cfg.defrost()
        # self._cfg.federate.make_global_eval = False
        # self._cfg.freeze(inform=False, save=False, check_cfg=False)

        # transport other enhanced args
        self.trainer.get_cur_state = self.get_cur_state

    def get_cur_state(self):
        return self.state

    def no_broadcast_evaluation_in_clients(self, msg_type='no_broadcast_evaluate_no_ft'):
        # no_broadcast_evaluate_no_ft and no_broadcast_evaluate_after_ft
        """
        modified from federatedscope.core.workers.server.Server.broadcast_model_para
        """
        assert msg_type in ["no_broadcast_evaluate_no_ft", "no_broadcast_evaluate_after_ft"]

        # no model_para is broadcast.
        receiver = list(self.comm_manager.neighbors.keys())

        # We define the evaluation happens at the begin of server behavior
        rnd = self.state  # important!

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=None))

    def broadcast_evaluation_in_clients(self, msg_type='evaluate_no_ft'):
        # evaluate_no_ft and evaluate_after_ft
        """
        modified from federatedscope.core.workers.server.Server.broadcast_model_para
        """
        assert msg_type in ["evaluate_no_ft", "evaluate_after_ft"]
        # broadcast to all clients
        receiver = list(self.comm_manager.neighbors.keys())

        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.models[0].state_dict()

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1  # important!

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))

    def terminate(self, msg_type='finish'):
        self.is_finish = True

        self._monitor.finish_fl()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=None))


class EnhanceClient(Client):
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

        cfg = config.clone()

        # save client_config.yaml begin!
        from pathlib import Path
        Path(cfg.outdir + '/configs').mkdir(parents=True, exist_ok=True)
        Path(cfg.outdir + '/checkpoints').mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.outdir, 'configs', f"client{ID}_config.yaml"), 'w') as outfile:
            from contextlib import redirect_stdout
            with redirect_stdout(outfile):
                tmp_cfg = copy.deepcopy(cfg)
                tmp_cfg.clear_aux_info()
                print(tmp_cfg.dump())
        # save client_config.yaml end!

        # load pretrained checkpoint!
        assert model is not None
        if cfg.model.type.startswith("attentive") and cfg.model.pretrain != "":
            supernet_cfg = cfg.model.clone()
            supernet_cfg.defrost()
            supernet_cfg.type = "attentive_supernet"
            supernet = get_model(supernet_cfg, local_data=None, backend=cfg.backend)
            # 加载supernet模型
            saved_state_dict = torch.load(supernet_cfg.pretrain, map_location=torch.device("cpu"))
            if "model" in saved_state_dict:
                saved_state_dict = saved_state_dict["model"]
            if "state_dict" in saved_state_dict:
                saved_state_dict = saved_state_dict["state_dict"]
            supernet.load_state_dict(saved_state_dict, strict=True)

            # supernet中采样模型加载权重
            if cfg.model.type == "attentive_min_subnet":
                supernet.sample_min_subnet()
                pretrained = supernet.get_active_subnet(preserve_weight=True).to(device)
                torch.optim.swa_utils.update_bn(data['server'], pretrained, device=device)  # recalibrate_bn
            elif cfg.model.type == "attentive_subnet":
                assert hasattr(cfg.model, 'arch_cfg')
                supernet.set_active_subnet(**cfg.model.arch_cfg)
                pretrained = supernet.get_active_subnet(preserve_weight=True).to(device)
                torch.optim.swa_utils.update_bn(data['server'], pretrained, device=device)  # recalibrate_bn
            else:  # "attentive_supernet"
                pretrained = supernet

            model.load_state_dict(pretrained.state_dict(), strict=True)
            print(f"Client{ID} Loaded pretrained checkpoint.")
            del supernet

        super(EnhanceClient, self).__init__(ID, server_id, state, cfg, data, model, device, strategy, is_unseen_client,
                                            *args, **kwargs)

        self.register_handlers('no_broadcast_evaluate_no_ft', self.callback_funcs_for_evaluate_no_finetune,
                               ['metrics'])
        self.register_handlers('evaluate_no_ft', self.callback_funcs_for_evaluate_no_finetune,
                               ['metrics'])
        self.register_handlers('no_broadcast_evaluate_after_ft', self.callback_funcs_for_evaluate_after_finetune,
                               ['metrics'])
        self.register_handlers('evaluate_after_ft', self.callback_funcs_for_evaluate_after_finetune,
                               ['metrics'])

        # transport other enhanced args
        self.trainer.get_cur_state = self.get_cur_state

    def get_cur_state(self):
        return self.state

    def callback_funcs_for_evaluate_no_finetune(self, message: Message):
        """
        modified from federatedscope.core.workers.client.Client.callback_funcs_for_evaluate
        Specifically, remove finetune and add logger.
        - new! save current client models.
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state

        if message.content is not None:  # evaluate_no_ft
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
        else:  # no_broadcast_evaluate_no_ft,
            # save client model
            torch.save({'round': self.state, 'model': self.model.state_dict()},
                       os.path.join(self._cfg.outdir, 'checkpoints', f"client{self.ID}_model.pth"))

        # if self._cfg.finetune.before_eval:

        metrics = {}
        for split in self._cfg.eval.split:
            # TODO: The time cost of evaluation is not considered here
            eval_metrics = self.trainer.evaluate(
                target_data_split_name=split)

            metrics.update(**eval_metrics)

        formatted_eval_res = self._monitor.format_eval_res(
            metrics,
            rnd=self.state,
            role=f'Client #{self.ID}({message.msg_type})',
            forms=['raw'],
            return_raw=True)
        logger.info(formatted_eval_res)  # NOTE(Variant): add this to output

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_evaluate_after_finetune(self, message: Message):
        """
        modified from federatedscope.core.workers.client.Client.callback_funcs_for_evaluate
        Specifically, add logger only.
        - new! save finetuned client models.
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state

        if message.content is not None:  # evaluate_after_ft
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
        else:  # no_broadcast_evaluate_after_ft
            pass

        metrics = {}
        if self._cfg.finetune.before_eval:
            sample_size, _, results = self.trainer.finetune()

            finetune_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role=f'Client #{self.ID}(w/ FT)({message.msg_type})',
                return_raw=True)
            logger.info(finetune_log_res)  # NOTE(Variant): add this to output

            # save client model
            torch.save({'round': self.state, 'model': self.model.state_dict()},
                       os.path.join(self._cfg.outdir, 'checkpoints', f"client{self.ID}_finetune_model.pth"))

        for split in self._cfg.eval.split:
            # TODO: The time cost of evaluation is not considered here
            eval_metrics = self.trainer.evaluate(
                target_data_split_name=split)
            metrics.update(**eval_metrics)

        formatted_eval_res = self._monitor.format_eval_res(
            metrics,
            rnd=self.state,
            role=f'Client #{self.ID}(w/ FT)({message.msg_type})',
            forms=['raw'],
            return_raw=True)
        logger.info(formatted_eval_res)  # NOTE(Variant): add this to output

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_model_para(self, message: Message):
        """
        simplified federatedscope.core.workers.client.Client.callback_funcs_for_model_para
        """
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        self.trainer.update(content, strict=True)
        self.state = round

        sample_size, model_para_all, results = self.trainer.train()

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
                    content=(sample_size, model_para_all)))

    def callback_funcs_for_finish(self, message: Message):
        logger.info(f"================= client {self.ID} received finish message =================")
        self._monitor.finish_fl()


def call_enhance_fl_worker(method):
    if method == 'BaseFL' or method == 'basefl':
        worker_builder = {'client': EnhanceClient, 'server': EnhanceServer}
        return worker_builder


register_worker('BaseFL', call_enhance_fl_worker)
