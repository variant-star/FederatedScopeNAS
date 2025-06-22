import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

from matplotlib import font_manager

font_path = "C:/Windows/Fonts/simsun.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False

with open("../exp-standalone-runs/cifar100/series_19subnets_infos.json", "r") as f:
    ground_truth = json.load(f)

with open("../exp-standalone-runs/cifar100/series_19subnets_infos_based_20231115205315.json", "r") as f:
    resource_not_aware = json.load(f)

with open("../exp-standalone-runs/cifar100/series_19subnets_infos_based_20231207103012.json", "r") as f:
    resource_aware = json.load(f)

with open("../exp-standalone-runs/cifar100/series_19subnets_infos_based_20231125134546.json", "r") as f:
    hetero_agnostic = json.load(f)


def normalize(A):
    return (A - np.mean(A)) / np.std(A)


last_acc = np.array([_["last_acc"] for _ in ground_truth])
idx_sorted = np.argsort(last_acc)

last_acc = normalize(last_acc)[idx_sorted]
last_acc = np.argsort(last_acc) + 1

# FLOPS
FLOPs = np.array([_["flops"] for _ in ground_truth])
FLOPs = normalize(FLOPs)[idx_sorted]
FLOPs = np.argsort(FLOPs) + 1
# PARAMS
params = np.array([_["params"] for _ in ground_truth])
params = normalize(params)[idx_sorted]
params = np.argsort(params) + 1

# Three supernet training status
proposed = np.array([-_["supernet_test_loss"] for _ in hetero_agnostic])
proposed = normalize(proposed)[idx_sorted]
proposed = np.argsort(proposed) + 1

resource_aware_status = np.array([-_["supernet_test_loss"] for _ in resource_aware])
resource_aware_status = normalize(resource_aware_status)[idx_sorted]
resource_aware_status = np.argsort(resource_aware_status) + 1

resource_not_aware_status = np.array([-_["supernet_test_loss"] for _ in resource_not_aware])
resource_not_aware_status = normalize(resource_not_aware_status)[idx_sorted]
resource_not_aware_status = np.argsort(resource_not_aware_status) + 1

# start plotting

colors = sns.color_palette("hls", 5)

variants = ['proposed', 'resource_aware_status', 'resource_not_aware_status', 'FLOPs', 'params', ]
# map_to_title = {
#     'proposed': 'resource-resilient federated training',
#     'FLOPs': 'FLOPs-based',
#     'params': '#params-based',
#     'resource_aware_status': 'resource-aware federated training',
#     'resource_not_aware_status': 'resource-agnostic federated training'
# }

map_to_title = {
    'proposed': '资源弹性联邦超网络训练策略',
    'FLOPs': '基于FLOPs',
    'params': '基于#Params',
    'resource_aware_status': '资源感知联邦超网络训练策略',
    'resource_not_aware_status': '资源无关联邦超网络训练策略'
}

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(17.5, 3.7))

for i in range(len(variants)):
    ax = axes[i]
    metric = eval(variants[i])
    rho = spearmanr(metric, last_acc)[0]
    tau = kendalltau(metric, last_acc)[0]
    ax.scatter(last_acc, metric, label=r"$\rho:$" + f"{rho:.4f}" + "  " + r"$\tau:$" + f"{tau:.4f}", color=colors[i]) #
    ax.plot(last_acc, metric, color=colors[i])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
    ax.set_xticks(np.arange(1, 19.5, 2))
    ax.set_yticks(np.arange(1, 19.5, 2))
    ax.set_xlim(0.9, 19.1)
    ax.set_ylim(0.9, 19.1)
    # ax.axis('equal')
    # ax.set_aspect('equal')
    ax.set_ylabel("代理排序")
    ax.set_xlabel("真实排序")
    ax.set_title(map_to_title[variants[i]])
    ax.legend(handlelength=0, handleheight=0, loc="upper left")

# plt.subplots_adjust(top=0.95, bottom=0.1, right=0.99, left=0.05, hspace=0.3, wspace=0.25)
plt.tight_layout()
# plt.show()
# plt.savefig("training_stability.pdf", bbox_inches='tight', pad_inches=0)
plt.savefig("D:/OneDrive - zju.edu.cn/VisualStudioRepos/zjuthesis-nas/figure/chapter4/yang8.pdf", bbox_inches='tight', pad_inches=0.05, format='pdf')
