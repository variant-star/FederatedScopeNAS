from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau

from predictor import JsonData, RFPredictor, NeuralPredictor
from predictor import get_imbalance


if __name__ == "__main__":
    # 获取
    CLIENT_I = 1
    PRED_METHOD = 'dl'  # choice: ['ml', 'dl']
    IMBALANCE_INFO = None  # ['gini', 'kldiv', 'shannon', 'cir']

    imbalance_info = None
    if IMBALANCE_INFO is not None:
        # 加载联邦数据集
        from federatedscope.core.configs.config import global_cfg
        init_cfg = global_cfg.clone()
        init_cfg.set_new_allowed(True)  # to receive new config dict
        init_cfg.merge_from_file("federatedscope/main_nas.yaml")

        from federatedscope.core.auxiliaries.utils import setup_seed
        setup_seed(init_cfg.seed)

        from federatedscope.contrib.auxiliaries.seed_data_builder import get_seed_data
        fs_data, _ = get_seed_data(config=init_cfg.clone(), client_cfgs=None)

        imbalance_info = get_imbalance(
            fs_data[CLIENT_I].train_data,
            n_classes=10 if init_cfg.data.type.lower() == 'cifar10' else 100,
            imbalance_metric=IMBALANCE_INFO
        )

    raw_data = JsonData(
        f"nas_fl_lr_on_cifar100_lr0.1_lstep1/sub_exp_20231024174102/client{CLIENT_I}_serverpopu2000_infos.json",
        imbalance_info=imbalance_info)

    # build machine learning dataset
    X, Y, Y_aux, target_names, aux_target_names = raw_data.get_ml_dataset(target_names=['acc'],
                                                                          aux_target_names=['flops', 'params'])
    X_train, X_test, Y_train, Y_test, Y_aux_train, Y_aux_test = train_test_split(X, Y, Y_aux,
                                                                                 test_size=0.9, random_state=42)

    if PRED_METHOD == "ml":
        # build machine learning predictor
        predictor = RFPredictor(max_depth=15, max_features='auto', criterion="mse", random_state=0)
        predictor.fit(X_train, Y_train)
        Y_pred = predictor.predict(X_test)

    elif PRED_METHOD == "dl":
        # build deep learning predictor
        predictor = NeuralPredictor(in_dim=raw_data.arch_dim + raw_data.extra_dim)
        # predictor.pretrain(X_train, Y_aux_train, X_test, Y_aux_test, aux_target_names=aux_target_names, max_epoch=40, lr=0.01)
        predictor.fit(X_train, Y_train, X_test, Y_test, target_names=target_names, max_epoch=40, lr=0.01)
        Y_pred = predictor.predict(X_test)

    else:
        raise ValueError(f"Unknown PRED_METHOD: {PRED_METHOD}")

    # start plotting
    fig = plt.figure(figsize=(8, 4))

    ax = plt.subplot(121)
    x_data, y_data = Y_aux_test[:, aux_target_names.index("flops")], Y_test
    ax.scatter(x_data, y_data)
    ax.set_xlabel("flops")
    ax.set_ylabel("loss")
    ax.set_title(f"pearson: {pearsonr(x_data, y_data)[0]:.4f}, kendall: {kendalltau(x_data, y_data)[0]:.4f}")

    ax = plt.subplot(122)
    x_data, y_data = Y_pred, Y_test
    ax.scatter(x_data, y_data)
    ax.set_xlabel("pred")
    ax.set_ylabel("loss")
    ax.set_title(f"pearson: {pearsonr(x_data, y_data)[0]:.4f}, kendall: {kendalltau(x_data, y_data)[0]:.4f}")

    plt.tight_layout()
    plt.show()