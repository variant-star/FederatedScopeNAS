import os
import json
import torch
import torchvision
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, kendalltau

from standalone_train import config_args, config_random, prepare_dataloader
from federatedscope.contrib.model.attentive_net import call_attentive_net

from train_val import train_one_epoch, evaluate_one_epoch


# # CIFAR10
# base_path = "exp/NAS_attentive_supernet_on_cifar10_lr0.1_lstep5/sub_exp_20231128020710" # (pear: 0.3667, tau: 0.3086)
# base_path = "exp/NAS_attentive_supernet_on_cifar10_lr0.1_lstep5/sub_exp_20231128020716" # (pear: 0.2536, tau: 0.2544)
# base_path = "exp/NAS_attentive_supernet_on_cifar10_lr0.1_lstep5/sub_exp_20231129232206" # (pear: 0.1996, tau: 0.1006)
# base_path = "exp/NAS_attentive_supernet_on_cifar10_lr0.1_lstep5/sub_exp_20231201103748" # (pear: 0.3592, tau: 0.2412)
# # One-Shot Training
# base_path = "exp/OneShot_attentive_supernet_on_cifar10_lr0.1_lstep1/sub_exp_20231207213540" # (pear: -0.0789, tau: -0.0944)

# # CIFAR100
# base_path = "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231115095919"  # pear: 0.1223, tau: 0.0899 (pear: -0.3787, tau: -0.3136)
# base_path = "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231116091212"  # pear: 0.4548, tau: 0.2000 (pear: 0.2921, tau: 0.2478)
# base_path = "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231116222209"  # pear: 0.3733, tau: 0.2889 (pear: 0.5054, tau: 0.3918)
# base_path = "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117085005"  # pear: 0.2203, tau: 0.3410 (pear: 0.4394, tau: 0.3588)
# base_path = "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117114505" # pear: 0.2629, tau: 0.2444 (pear: 0.3420, tau: 0.2529)
# base_path = "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231125134546" # pear: 0.4116, tau: 0.4222 (pear: 0.6675, tau: 0.4529)
# base_path = "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231130224605" # pear: 0.5071, tau: 0.3596 (pear: 0.2521, tau: 0.1930)
# # One-Shot Training
base_path = "exp/OneShot_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231115205315" # pear: 0.1143, tau: 0.2444 (pear: 0.7228, tau: 0.5088)
# base_path = "exp/OneShot_attentive_supernet_on_cifar100_lr0.1_lstep1/sub_exp_20231207103012" # (pear: -0.0797, tau: -0.0877)

# # # TinyImageNet
# base_path = "exp/NAS_attentive_supernet_on_tinyimagenet_lr0.1_lstep5/sub_exp_20231129093240"  # (pear: -0.0945, tau: -0.0412)
# base_path = "exp/NAS_attentive_supernet_on_tinyimagenet_lr0.1_lstep5/sub_exp_20231129175546"  # (pear: 0.2895, tau: 0.3068)
# base_path = "exp/NAS_attentive_supernet_on_tinyimagenet_lr0.1_lstep5/sub_exp_20231130203810"  # (pear: 0.2664, tau: 0.1941)
# base_path = "exp/NAS_attentive_supernet_on_tinyimagenet_lr0.1_lstep5/sub_exp_20231201170131"  # (pear: 0.1134, tau: 0.0360)
# # One-Shot
# base_path = "exp/OneShot_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231208100713"  # (pear: 0.1756, tau: 0.1353)
# base_path = "exp/OneShot_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20231209100404"  # (pear: 0.1439, tau: 0.1716)

with open("exp-standalone-runs/cifar100/series_19subnets_infos.json", "r") as f:
    subnets_infos = json.load(f)


if __name__ == "__main__":
    args = config_args()
    config_random(args)
    args.device = torch.device("cuda:0")

    train_dataloader, val_dataloader = prepare_dataloader(args)

    # 创建supernet
    args.type = "attentive_supernet"
    supernet = call_attentive_net(args, input_shape=None).to(args.device)
    args.type = "attentive_subnet"

    saved_state_dict = torch.load(os.path.join(base_path, "checkpoints", "supernet.pth"))["model"]
    supernet.load_state_dict(saved_state_dict, strict=True)

    for subnet_info in subnets_infos:
        supernet.set_active_subnet(**subnet_info["arch_cfg"])
        subnet = supernet.get_active_subnet(preserve_weight=True).to(args.device)
        torch.optim.swa_utils.update_bn(train_dataloader, subnet, device=args.device)
        test_loss, test_acc, _ = evaluate_one_epoch(subnet, val_dataloader, use_amp=True, device=args.device)
        subnet_info.update({"supernet_test_loss": test_loss, "supernet_test_acc": test_acc})

    truth_accs = [subnet_info["last_acc"] for subnet_info in subnets_infos]
    print(truth_accs)
    supernet_accs = [subnet_info["supernet_test_loss"] for subnet_info in subnets_infos]
    print(supernet_accs)

    print(f"pear: {pearsonr(truth_accs, supernet_accs)[0]:.4f}, "
          f"tau: {kendalltau(truth_accs, supernet_accs)[0]:.4f}")

    with open("exp-standalone-runs/cifar100/series_19subnets_infos_based_20231115205315_tmp.json", "w") as f:
        json.dump(subnets_infos, f, indent=4)




#
#
# # with open("/server_popu1000_infos.json") as f:
# #     infos = json.load(f)
#
#
# flops = [subnet_info['flops']/1e6 for subnet_info in infos]
# acc = [subnet_info['server_recalibrate_bn_acc'] for subnet_info in infos]
# loss = [subnet_info['server_recalibrate_bn_loss'] for subnet_info in infos]
#
#
# fig = plt.figure(figsize=(8, 4))
# ax = plt.subplot(121)
# ax.scatter(flops, loss)
# ax.set_xlabel("FLOPs")
# ax.set_ylabel("loss")
#
# ax = plt.subplot(122)
# ax.scatter(flops, acc)
# ax.set_xlabel("FLOPs")
# ax.set_ylabel("acc")
#
# plt.tight_layout()
# plt.show()