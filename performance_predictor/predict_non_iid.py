from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau

from predictor import JsonData, RFPredictor, NeuralPredictor
from predictor import get_imbalance

SUB_PATH = "../exp/nas_fl_lr_on_cifar100_lr0.1_lstep1/sub_exp_20231031114136"

# 获取
CLIENT_NUM = 8
PRED_METHOD = 'dl'  # choice: ['ml', 'dl']

if __name__ == "__main__":
    # make containers to store training set or validation set.----------------------------------------------------------

    raw_data = {}
    X, Y, Y_aux = {}, {}, {}
    X_train, X_test, Y_train, Y_test, Y_aux_train, Y_aux_test = {}, {}, {}, {}, {}, {}

    for cid in range(CLIENT_NUM+1):  # "cid == 0" means server client.
        raw_data = JsonData(
            f"{SUB_PATH}/server_popu_infos.json" if cid == 0 else f"{SUB_PATH}/client{cid}_serverpopu_infos.json",
            imbalance_info=None)

        # build machine learning dataset
        X[cid], Y[cid], Y_aux[cid], target_names, aux_target_names = raw_data.get_ml_dataset(target_names=['loss'],
                                                                                             aux_target_names=['flops', 'params'])
        X_train[cid], X_test[cid], Y_train[cid], Y_test[cid], Y_aux_train[cid], Y_aux_test[cid] \
            = train_test_split(X[cid], Y[cid], Y_aux[cid], test_size=0.9, random_state=42)

    # build deep learning performance predictor-------------------------------------------------------------------------

    # predictor = NeuralPredictor(in_dim=raw_data.arch_dim + raw_data.extra_dim)
    # predictor.pretrain(X_train, Y_aux_train, X_test, Y_aux_test, aux_target_names=aux_target_names,
    #                    max_epoch=40, lr=0.01)
    # predictor.fit(X_train, Y_train, X_test, Y_test, target_names=target_names, max_epoch=40, lr=0.01)
    # Y_pred = predictor.predict(X_test)

    # build machine learning performance predictor----------------------------------------------------------------------

    predictor = {}
    for cid in range(CLIENT_NUM+1):
        predictor[cid] = RFPredictor(max_depth=15, max_features='auto', criterion="squared_error", random_state=0)
        predictor[cid].fit(X_train[cid], Y_train[cid])
        # Y_pred = predictor[cid].predict(X_test)

    # start plotting----------------------------------------------------------------------------------------------------

    fig, axes = plt.subplots(nrows=CLIENT_NUM+1, ncols=CLIENT_NUM+1, figsize=(38, 32))

    for base_cid in range(CLIENT_NUM+1):
        for target_cid in range(CLIENT_NUM+1):

            ax = axes[base_cid, target_cid]

            y_pred_target, y_truth_target = predictor[base_cid].predict(X_test[target_cid]), Y_test[target_cid]
            ax.scatter(y_pred_target, y_truth_target)
            if base_cid == target_cid:
                y_pred_base, y_truth_base = predictor[base_cid].predict(X_train[base_cid]), Y_train[base_cid]
                ax.scatter(y_pred_base, y_truth_base, color="red")  # 避免遮盖
                ax.scatter(y_truth_base, y_truth_base, color="green")  # 避免遮盖
            ax.set_xlabel(f"pred({'server' if base_cid == 0 else 'client'}{base_cid} predictor)")
            ax.set_ylabel(f"loss({'server' if target_cid == 0 else 'client'}{target_cid})")
            ax.set_title(f"pearson: {pearsonr(y_pred_target, y_truth_target)[0]:.4f}, "
                         f"kendall: {kendalltau(y_pred_target, y_truth_target)[0]:.4f}")

    plt.tight_layout()
    plt.show()