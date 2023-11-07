import logging
import copy
import os

from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your worker here.
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
        Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.outdir, "server_config.yaml"), 'w') as outfile:
            from contextlib import redirect_stdout
            with redirect_stdout(outfile):
                tmp_cfg = copy.deepcopy(cfg)
                tmp_cfg.clear_aux_info()
                print(tmp_cfg.dump())
        # save server_config.yaml end!

        super(EnhanceServer, self).__init__(ID, state, cfg, data, model, client_num, total_round_num, device, strategy,
                                            unseen_clients_id, **kwargs)

        # transport other enhanced args
        self.trainer.get_cur_state = self.get_cur_state

    def get_cur_state(self):
        return self.state


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
        Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.outdir, f"client{ID}_config.yaml"), 'w') as outfile:
            from contextlib import redirect_stdout
            with redirect_stdout(outfile):
                tmp_cfg = copy.deepcopy(cfg)
                tmp_cfg.clear_aux_info()
                print(tmp_cfg.dump())
        # save client_config.yaml end!

        super(EnhanceClient, self).__init__(ID, server_id, state, cfg, data, model, device, strategy, is_unseen_client,
                                            *args, **kwargs)

        # transport other enhanced args
        self.trainer.get_cur_state = self.get_cur_state

    def get_cur_state(self):
        return self.state


def call_enhance_fl_worker(method):
    if method == 'BaseFL' or method == 'basefl':
        worker_builder = {'client': EnhanceClient, 'server': EnhanceServer}
        return worker_builder


register_worker('BaseFL', call_enhance_fl_worker)
