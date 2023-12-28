import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau

from predictor import JsonData, RFPredictor, NeuralPredictor
from predictor import get_imbalance

SUB_PATH = "../exp/nas_fl_lr_on_cifar100_lr0.1_lstep1/sub_exp_20231031114136"

# 获取
CLIENT_NUM = 8
PRED_METHOD = 'dl'  # choice: ['ml', 'dl']
IMBALANCE_INFO = ['gini', 'kldiv', 'std']  # 'kldiv', 'cir'

if __name__ == "__main__":

    # 加载联邦数据集
    from federatedscope.core.configs.config import global_cfg
    init_cfg = global_cfg.clone()
    init_cfg.set_new_allowed(True)  # to receive new config dict
    init_cfg.merge_from_file("../nas.yaml")

    from federatedscope.core.auxiliaries.utils import setup_seed
    setup_seed(init_cfg.seed)

    from federatedscope.contrib.auxiliaries.seed_data_builder import get_seed_data
    # fs_data, _ = get_seed_data(config=init_cfg.clone(), client_cfgs=None)
    df = pd.read_csv(f"{SUB_PATH}/data_reports.csv", delimiter=",", skipinitialspace=True)
    df = df.set_index("node####")

    # make containers to store training set or validation set.----------------------------------------------------------

    raw_data = {}
    X, Y, Y_aux = {}, {}, {}
    X_train, X_test, Y_train, Y_test, Y_aux_train, Y_aux_test = {}, {}, {}, {}, {}, {}

    for cid in range(CLIENT_NUM+1):  # "cid == 0" means server client.
        imbalance_info = get_imbalance(
            np.array(df.loc[f"{'Server' if cid == 0 else 'Client'}#{cid}#train"])[:-1],  # fs_data[cid].train_data,
            n_classes=10 if init_cfg.data.type.lower() == 'cifar10' else 100,
            imbalance_metric=IMBALANCE_INFO
        )
        raw_data = JsonData(
            f"{SUB_PATH}/server_popu_infos.json" if cid == 0 else f"{SUB_PATH}/client{cid}_serverpopu_infos.json",
            imbalance_info=imbalance_info)

        # build machine learning dataset
        X[cid], Y[cid], Y_aux[cid], target_names, aux_target_names = raw_data.get_ml_dataset(target_names=['loss'],
                                                                                             aux_target_names=['flops', 'params'])
        X_train[cid], X_test[cid], Y_train[cid], Y_test[cid], Y_aux_train[cid], Y_aux_test[cid] \
            = train_test_split(X[cid], Y[cid], Y_aux[cid], test_size=0.95, random_state=42)

    # build machine learning performance predictor----------------------------------------------------------------------
    INCLUDE_CID = [0, 1, 2, 3, 4]  # 用于训练
    EXCLUDE_CID = [5, 6, 7, 8]  # 用于测试（性能并不好）

    predictor = RFPredictor(max_depth=15, max_features='auto', criterion="squared_error", random_state=0)
    predictor.fit(np.vstack([X_train[cid] for cid in INCLUDE_CID]), np.hstack([Y_train[cid] for cid in INCLUDE_CID]))
    # Y_pred = predictor[cid].predict(X_test)

    # start plotting----------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax = axes

    for cid in INCLUDE_CID:
        y_pred_base, y_truth_base = predictor.predict(X_test[cid]), Y_test[cid]
        ax.scatter(y_pred_base, y_truth_base,
                   label=f"TRAIN({cid}) | "
                         f"pear: {pearsonr(y_pred_base, y_truth_base)[0]:.4f}, "
                         f"tau: {kendalltau(y_pred_base, y_truth_base)[0]:.4f}")

    for cid in EXCLUDE_CID:
        y_pred_target, y_truth_target = predictor.predict(X_test[cid]), Y_test[cid]
        ax.scatter(y_pred_target, y_truth_target,
                   label=f"TEST({cid}) | "
                         f"pear: {pearsonr(y_pred_target, y_truth_target)[0]:.4f}, "
                         f"tau: {kendalltau(y_pred_target, y_truth_target)[0]:.4f}")

    ax.legend()

    plt.tight_layout()
    plt.show()

    # start plotting----------------------------------------------------------------------------------------------------

    # fig, axes = plt.subplots(nrows=CLIENT_NUM+1, ncols=CLIENT_NUM+1, figsize=(36, 32))
    #
    # for base_cid in range(CLIENT_NUM+1):
    #     for target_cid in range(CLIENT_NUM+1):
    #
    #         ax = axes[base_cid, target_cid]
    #
    #         x_data, y_data = predictor[base_cid].predict(X_test[target_cid]), Y_test[target_cid]
    #         ax.scatter(x_data, y_data)
    #         ax.set_xlabel(f"pred({'server' if base_cid == 0 else 'client'}{base_cid} predictor)")
    #         ax.set_ylabel(f"loss({'server' if target_cid == 0 else 'client'}{target_cid})")
    #         ax.set_title(f"pearson: {pearsonr(x_data, y_data)[0]:.4f}, kendall: {kendalltau(x_data, y_data)[0]:.4f}")
    #
    # plt.tight_layout()
    # plt.show()