import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager

font_path = "C:/Windows/Fonts/simsun.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 6))

for i, DATASET in enumerate(["CIFAR10", "CIFAR100", "TinyImageNet"]):
    for j, N_CLIENTS in enumerate([8, 12, 16, 20]):

        ax = axes[i, j]

        df = pd.read_csv(f"{DATASET}_c{N_CLIENTS}_data_reports.csv")
        df = df.set_index('node####')

        df_train = df.iloc[3::3, :-1]
        df_test = df.iloc[4::3, :-1]

        row_index = [str(i) for i in range(1, N_CLIENTS + 1)]
        df_train.rename(index=dict(zip(df_train.index, row_index)), inplace=True)
        df_test.rename(index=dict(zip(df_test.index, row_index)), inplace=True)

        df = df_train + df_test

        df.plot(kind='barh', stacked=True, ax=ax)  # 绘制堆叠直方图
        # ax.set_ylabel(f'Client')
        ax.set_ylabel(f'客户端')
        # ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_ylabel('')
        ax.legend("", frameon=False)  # 添加图例
        if i == 2:
            # ax.set_xlabel("number of samples")
            ax.set_xlabel("样本数")


# plt.tight_layout()
plt.subplots_adjust(top=0.995, bottom=0.05, right=0.995, left=0.015, hspace=0.15, wspace=0.1)
# plt.show()
# plt.savefig("data_distributions.pdf", bbox_inches='tight', pad_inches=0, format='pdf')
plt.savefig("D:/OneDrive - zju.edu.cn/VisualStudioRepos/zjuthesis-nas/figure/chapter4/yang7.pdf", bbox_inches='tight', pad_inches=0, dpi=600, format='pdf')
