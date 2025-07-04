import numpy as np
total_round = 60

for cur_round in range(0, total_round):
    accs = []
    with open("exp/FedAgg_attentive_supernet_on_tinyimagenet_lr0.1_lstep1/sub_exp_20240122104215/exp_print.log", "r") as f:
        while True:
            line = f.readline()
            if line:
                if f"(no_broadcast_evaluate_no_ft)', 'Round': {cur_round}," in line:
                    test_acc = eval(line.split('INFO: ')[-1])['Results_raw']['test_acc']
                    accs.append(test_acc)
            else:
                break

    print(f"Client_num: {len(accs)}, round:{cur_round}, mean_acc: {np.mean(accs)}")