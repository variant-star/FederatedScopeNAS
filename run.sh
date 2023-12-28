screen -ls|awk 'NR>=2&&NR<=20{print $1}'|awk '{print "screen -S "$1" -X quit"}'|sh

# TinyImageNet          featurize dataset download 073a9ce1-f404-42c8-b067-27b730c2fd73

# NAS train supernet
python main.py --cfg yamls/nas_server_cfg.yaml --client_cfg yamls/nas_client_cfg.yaml \
  federate.client_num 12 \
  federate.total_round_num 120 \
  data.type cifar10

# NAS train supernet (oneshot)
python main.py --cfg yamls/oneshot_cfg.yaml federate.client_num 8 federate.total_round_num 120 data.type cifar100
python main.py --cfg yamls/oneshot_cfg.yaml federate.client_num 8 federate.total_round_num 120 data.type cifar10
python main.py --cfg yamls/oneshot_cfg.yaml federate.client_num 8 federate.total_round_num 120 data.type tinyimagenet
python main.py --cfg yamls/oneshot_cfg.yaml federate.client_num 20 federate.total_round_num 120 data.type cifar100

# search  # 已修改为更为简单的调用方法，无需添加额外参数
python search.py --cfg yamls/nas_server_cfg.yaml --client_cfg yamls/nas_client_cfg.yaml \
  model.pretrain "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231125134546/checkpoints/supernet.pth" \
  federate.client_num 20

# retrain KEMF (distillation-based fusion)
python main.py --cfg yamls/kemf_server_cfg.yaml --client_cfg yamls/kemf_client_cfg.yaml \
  model.pretrain "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117114505/checkpoints/supernet.pth" \
  client_models_cfg "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117114505/client_best_infos.json" \
  federate.total_round_num 120 \
  federate.client_num 16

# retrain FedMD (regularized-based)
python main.py --cfg yamls/fedmd_server_cfg.yaml --client_cfg yamls/fedmd_client_cfg.yaml \
  model.pretrain "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117114505/checkpoints/supernet.pth" \
  client_models_cfg "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117114505/client_best_infos.json" \
  federate.total_round_num 120 \
  federate.client_num 16

# retrain FedAvg (heteroFedAvg: FedAgg)
python main.py --cfg yamls/fedagg_server_cfg.yaml --client_cfg yamls/fedagg_client_cfg.yaml \
  model.pretrain "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117114505/checkpoints/supernet.pth" \
  client_models_cfg "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117114505/client_best_infos.json" \
  federate.total_round_num 60 \
  federate.client_num 16

# retrain FedAvg (homoFedAvg)
python main.py --cfg yamls/all_min_server_cfg.yaml --client_cfg yamls/all_min_client_cfg.yaml \
  model.pretrain "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231117114505/checkpoints/supernet.pth" \
  federate.total_round_num 120 \
  federate.client_num 16

# 11.29

python main.py --cfg yamls/fedagg_server_cfg.yaml --client_cfg yamls/fedagg_client_cfg.yaml \
  model.pretrain "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231130224605/checkpoints/supernet.pth" \
  client_models_cfg "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231130224605/client_best_infos.json" \
  federate.total_round_num 60 \
  federate.client_num 12 \
  data.type cifar100

python main.py --cfg yamls/fedagg_server_cfg.yaml --client_cfg yamls/fedagg_client_cfg.yaml \
  model.pretrain "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231130224605/checkpoints/supernet.pth" \
  client_models_cfg "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231130224605/client_random_infos.json" \
  federate.total_round_num 60 \
  federate.client_num 12 \
  data.type cifar100

python main.py --cfg yamls/all_min_server_cfg.yaml --client_cfg yamls/all_min_client_cfg.yaml \
  model.pretrain "exp/NAS_attentive_supernet_on_cifar100_lr0.1_lstep5/sub_exp_20231130224605/checkpoints/supernet.pth" \
  federate.total_round_num 60 \
  federate.client_num 12 \
  data.type cifar100

