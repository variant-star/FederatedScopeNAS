import copy

import torch
import torch.nn as nn
import json
import numpy as np
from copy import deepcopy
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from AttentiveNAS.utils.progress import AverageMeter

from scipy.stats import entropy


class JsonData:
    def __init__(self, path, imbalance_info=None):
        self.inputs, self.targets = self.parse_data(path)
        self.inputs = np.array(self.inputs)

        self.n_samples, self.arch_dim = self.inputs.shape
        self.extra_dim = 0

        if imbalance_info is not None:
            imbalance_values = list(imbalance_info.values())
            imbalance_values = np.array([imbalance_values])
            imbalance_values = np.repeat(imbalance_values, repeats=self.n_samples, axis=0)
            self.inputs = np.hstack((self.inputs, imbalance_values))

            self.extra_dim = self.inputs.shape[-1] - self.arch_dim

    # <resolution><---------------width----------->  <-------depth------->  <-----kernel-------->  <-----expand-------->
    # {32,[32, 16, 24, 40, 64, 96, 160, 320, 1280], [1, 4, 4, 4, 4, 4, 1], [3, 5, 5, 5, 5, 5, 3], [1, 6, 6, 6, 3, 6, 6]}
    #  -   1    -   -   4   5   -   7    -     -     -- 11 12 13 14 15 --   -- 18 19 20 21 22 --   -- 25 26 27 -- -- --
    @staticmethod
    def cfg2vec(cfg):
        filtered_idx = [1, 4, 5, 7, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 25, 26, 27]
        res = [cfg['resolution']]
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            res += cfg[k]
        res = np.asarray(res, dtype=float)  # .reshape((1, -1))
        res = res / np.array([32, 32, 16, 24, 40, 64, 96, 160, 320, 1280, 1, 4, 4, 4, 4, 4, 1, 3, 5, 5, 5, 5, 5, 3, 1, 6, 6, 6, 3, 6, 6])
        res = res[filtered_idx]
        return res

    @staticmethod
    def parse_data(json_path):
        with open(json_path, 'r') as f:
            infos = json.load(f)
            infos.sort(key=lambda _: _['flops'])
            for info in infos:
                info['res'] = JsonData.cfg2vec(info['arch_cfg'])
        inputs = [_['res'] for _ in infos]
        # {flops: x00,000,000  params:x.000,000}
        targets = [
            {
                'flops': _['flops'],
                'params': _['params'],
                'loss': _['loss'],
                'acc': _['acc']
            }
            for _ in infos]
        return inputs, targets  # return type: List[numpy.array(int64)] List[Dict]

    def get_ml_dataset(self, target_names=None, aux_target_names=None):
        X = self.inputs
        if target_names is None:
            target_names = list(self.targets[0].keys())
        if aux_target_names is None:
            aux_target_names = list(self.targets[0].keys())
        Y = [[target[_] for _ in target_names] for target in self.targets]
        Y_aux = [[target[_] for _ in aux_target_names] for target in self.targets]
        return np.array(X), np.squeeze(np.array(Y)), np.squeeze(np.array(Y_aux)), target_names, aux_target_names
        # return type: np.array(np.array(int)), np.array(np.array(int)), np.array(np.array(int)), List[Str], List[Str]

    @staticmethod
    def convert_ml2dl(X, Y, target_names):
        return DLDataset(deepcopy(X), deepcopy(Y), deepcopy(target_names))


class DLDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, target_names):
        super(DLDataset, self).__init__()

        if targets.ndim == 1:
            targets = targets[:, np.newaxis]
        if "loss" in target_names:
            loss_index = target_names.index("loss")
            targets[:, loss_index] = targets[:, loss_index] * 1.0 / 0.004
        if "flops" in target_names:
            flops_index = target_names.index("flops")
            targets[:, flops_index] = (targets[:, flops_index] - 50_000_000) * 1.0 / (145_000_000 - 50_000_000)
        if "params" in target_names:
            params_index = target_names.index("params")
            targets[:, params_index] = (targets[:, params_index] - 2_165_420) * 1.0 / (5_100_804 - 2_165_420)

        self.inputs = torch.tensor(inputs, dtype=torch.float)
        self.targets = torch.tensor(np.squeeze(targets), dtype=torch.float)
        self.target_names = target_names

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class RFPredictor:
    def __init__(self, *args, **kwargs):
        super(RFPredictor, self).__init__()
        self.model = RandomForestRegressor(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def scratch_predict(self, cfg):
        cfg_vec = JsonData.cfg2vec(cfg)
        cfg_vec = np.array([cfg_vec])
        pred = self.predict(cfg_vec)[0]
        return pred


class MLP(nn.Module):
    def __init__(self, n_layers=3, in_dim=17, hidden_dim=400, out_dim=1):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # build mlp
        mlp = []
        for i in range(self.n_layers):
            mlp.append(
                nn.Sequential(
                    nn.Linear(
                        self.in_dim if i == 0 else self.hidden_dim,
                        self.hidden_dim,
                    ),
                    # nn.ReLU(inplace=True),
                    nn.Sigmoid(),
                )
            )
        self.mlp = nn.Sequential(*mlp)
        self.head = nn.Linear(self.hidden_dim, self.out_dim, bias=True)  # TODO(Variant): bias?
        # # NOTE(Variant): convert as transform function
        # self.base_value = nn.Parameter(
        #     torch.zeros(1, device=self.device), requires_grad=False
        # )

    def forward(self, x):
        return self.head(self.mlp(x))

    def change_head(self, n_dim=1):
        self.head = nn.Linear(self.hidden_dim, n_dim, bias=False)


class NeuralPredictor:
    def __init__(
            self,
            n_layers=3, in_dim=17, hidden_dim=400, out_dim=1,
    ):
        self.device = torch.device("cpu")
        self.model = MLP(n_layers, in_dim, hidden_dim, out_dim)

        self.out_names = None

    @staticmethod
    def train_val_pipeline(model, train_loader, test_loader, max_epoch, lr, device):
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-5, last_epoch=-1)

        for epoch_i in range(max_epoch):
            # train one epoch
            model.train()
            objs = AverageMeter('loss', '6.2f:')
            pbar = tqdm(train_loader)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logging
                n = inputs.size(0)
                objs.update(loss.item(), n)
                pbar.set_description(f'TRAIN Epoch[{epoch_i+1}/{max_epoch}]')
                pbar.set_postfix(loss=objs.avg)

            # validate per epoch
            if test_loader is not None:
                model.eval()
                objs = AverageMeter('loss', '6.2f:')
                pbar = tqdm(test_loader)
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(pbar):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        # logging
                        n = inputs.size(0)
                        objs.update(loss.item(), n)
                        pbar.set_description(f'TEST Epoch[{epoch_i+1}/{max_epoch}]')
                        pbar.set_postfix(loss=objs.avg)

            lr_scheduler.step()

    def pretrain(self, X_train, Y_aux_train, X_test=None, Y_aux_test=None, aux_target_names=None, max_epoch=0, lr=0):

        # transform to machine learning datasets and dataloaders
        train_aux_dataset = JsonData.convert_ml2dl(X_train, Y_aux_train, target_names=aux_target_names)
        train_aux_loader = torch.utils.data.DataLoader(train_aux_dataset, batch_size=32, shuffle=True)

        if X_test is not None and Y_aux_test is not None:
            test_aux_dataset = JsonData.convert_ml2dl(X_test, Y_aux_test, target_names=aux_target_names)
            test_aux_loader = torch.utils.data.DataLoader(test_aux_dataset, batch_size=32, shuffle=False)
        else:
            test_aux_loader = None

        # refactor the model
        self.model.change_head(len(aux_target_names))
        self.out_names = aux_target_names

        self.model.to(self.device)
        self.train_val_pipeline(self.model, train_aux_loader, test_aux_loader, max_epoch, lr, device=self.device)

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, target_names=None, max_epoch=0, lr=0):

        # transform to machine learning datasets and dataloaders
        train_dataset = JsonData.convert_ml2dl(X_train, Y_train, target_names=target_names)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        if X_test is not None and Y_test is not None:
            test_dataset = JsonData.convert_ml2dl(X_test, Y_test, target_names=target_names)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        else:
            test_loader = None

        # refactor the model
        self.model.change_head(len(target_names))
        self.out_names = target_names

        self.model.to(self.device)
        self.train_val_pipeline(self.model, train_loader, test_loader, max_epoch, lr, device=self.device)

    def predict(self, X_test):
        if not torch.is_tensor(X_test):
            X_test = torch.tensor(X_test, dtype=torch.float)
        X_test = X_test.to(self.device)

        self.model.eval()
        with torch.no_grad():
            Y_pred = self.model(X_test)
        Y_pred = Y_pred.cpu().numpy()

        if "loss" in self.out_names:
            loss_index = self.out_names.index("loss")
            Y_pred[:, loss_index] = Y_pred[:, loss_index] * 1.0 * 0.004
        if "flops" in self.out_names:
            flops_index = self.out_names.index("flops")
            Y_pred[:, flops_index] = Y_pred[:, flops_index] * 1.0 * (145_000_000 - 50_000_000) + 50_000_000
        if "params" in self.out_names:
            params_index = self.out_names.index("params")
            Y_pred[:, params_index] = Y_pred[:, params_index] * 1.0 * (5_100_804 - 2_165_420) + 2_165_420

        return np.squeeze(Y_pred)

    def scratch_predict(self, cfg):
        cfg_vec = JsonData.cfg2vec(cfg)
        cfg_vec = np.array([cfg_vec])
        pred = self.predict(cfg_vec)
        return pred.item()


def get_imbalance(imbalance_data, n_classes, imbalance_metric):  # input_type: "Dataset" or "Subset"
    indices = None
    while True:
        if hasattr(imbalance_data, "dataset"):
            indices = imbalance_data.indices[indices] if indices is not None else imbalance_data.indices
            imbalance_data = imbalance_data.dataset
        else:
            imbalance_targets = np.array(imbalance_data.targets)[indices]
            break
    imbalance_targets = imbalance_targets.tolist()

    cls_freq = np.bincount(imbalance_targets, minlength=n_classes)
    cls_prob = cls_freq * 1.0 / sum(cls_freq)

    metrics = {}
    if "kldiv" in imbalance_metric:
        metrics['kldiv'] = entropy(cls_prob, [1 / n_classes] * n_classes, base=2)
    if "shannon" in imbalance_metric:
        metrics['shannon'] = entropy(cls_prob, base=2)
    if "gini" in imbalance_metric:
        metrics['gini'] = 1 - sum(cls_prob ** 2)
    if "cir" in imbalance_metric:
        metrics['cir'] = max(cls_freq) * 1.0 / min(cls_freq)
    return metrics