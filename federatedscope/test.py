import torch
import torch.nn.functional as F
import math

from federatedscope.contrib.model.attentive_net import call_attentive_net
import re
import matplotlib.pyplot as plt
import numpy as np

import copy
from torch.cuda.amp import GradScaler, autocast


from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar, lifecycle

from federatedscope.contrib.auxiliaries.ensemble_related import calculate_ensemble_logits



# class AttrDict(dict):
#     __slots__ = ()
#     __getattr__ = dict.__getitem__
#     __setattr__ = dict.__setitem__
#
#
# supernet = call_attentive_net(AttrDict({"type": "attentive_supernet"}), None)
# for k, v in supernet.state_dict().items():
#     print(f"{k} | {v.size()}")
#
# min_subnet = call_attentive_net(AttrDict({"type": "attentive_min_subnet"}), None)
# # for k, v in min_subnet.state_dict().items():
# #     print(f"{k} | {v.size()}")
# from pprint import pprint
# pprint(min_subnet.config)


normal = {"max": [], "random": [], "min":[]}

with open("exp/nas_fl_attentive_min_subnet_on_cifar100_lr0.1_lstep1/exp_print.log", 'r') as f:
    while True:
        line = f.readline()
        if line:
            if "'Role': 'Server Supernet(" in line:
                acc = float(re.search(r"'test_acc': \d+.\d+", line).group()[12:])
                if "'Role': 'Server Supernet(max)" in line:
                    normal["max"].append(acc)
                elif "'Role': 'Server Supernet(random)" in line:
                    normal["random"].append(acc)
                elif "'Role': 'Server Supernet(min)" in line:
                    normal["min"].append(acc)
                else:
                    raise ValueError
        else:
            break

reverse = {"max": [], "random": [], "min":[]}

with open("exp/nas_fl_attentive_min_subnet_on_cifar100_lr0.1_lstep1/sub_exp_20230613143004/exp_print.log", 'r') as f:
    while True:
        line = f.readline()
        if line:
            if "'Role': 'Server Supernet(" in line:
                acc = float(re.search(r"'test_acc': \d+.\d+", line).group()[12:])
                if "'Role': 'Server Supernet(max)" in line:
                    reverse["max"].append(acc)
                elif "'Role': 'Server Supernet(random)" in line:
                    reverse["random"].append(acc)
                elif "'Role': 'Server Supernet(min)" in line:
                    reverse["min"].append(acc)
                else:
                    raise ValueError
        else:
            break


fig = plt.figure()

plt.plot(np.arange(len(normal['max'])), normal['max'], label="normal_max", color="red", linestyle='--')
plt.plot(np.arange(len(normal['min'])), normal['min'], label="normal_min", color="blue", linestyle='--')
plt.plot(np.arange(len(normal['random'])), normal['random'], label="normal_random", color="green", linestyle='--')
plt.plot(np.arange(len(reverse['max'])), reverse['max'], label="reverse_max", color="red")
plt.plot(np.arange(len(reverse['min'])), reverse['min'], label="reverse_min", color="blue")
plt.plot(np.arange(len(reverse['random'])), reverse['random'], label="reverse_random", color="green")

plt.legend()
plt.show()




