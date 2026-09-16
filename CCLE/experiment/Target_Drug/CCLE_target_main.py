# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
# from import_path import *
from CCLE.experiment.model import GModel
from CCLE.experiment.optimizer import Optimizer
from sklearn.model_selection import KFold
from sampler import Sampler
from CCLE.experiment.myutils import roc_auc, translate_result, common_data_index, dir_path

data_dir =dir_path(k=2) + "CCLE/processed_data/"
target_drug_cids = np.array([16038120])

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug.columns = cell_drug.columns.astype(int)
drug_cids = cell_drug.columns.values
cell_target_drug = np.array(cell_drug.loc[:, target_drug_cids], dtype=np.float32)
adj_mat_coo_data = sp.coo_matrix(cell_target_drug).data
cell_drug = np.array(cell_drug, dtype=np.float32)

target_indexes = common_data_index(drug_cids, target_drug_cids)

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)

# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)

# 加载细胞系-cna特征矩阵
cna = pd.read_csv(data_dir + "cna_feature.csv", index_col=0, header=0)
cna = np.array(cna, dtype=np.float32)

# 加载细胞系-mutaion特征矩阵
mutation = pd.read_csv(data_dir + "mutation_feature.csv", index_col=0, header=0)
mutation = np.array(mutation, dtype=np.float32)
mutation = (mutation != 0).astype(np.float32)
null_mask = np.zeros(cell_drug.shape, dtype=np.float32)

epochs = []
true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=21)

import os
# 自动创建结果保存目录（如果不存在）
os.makedirs("./result_data", exist_ok=True)
n_kfold = 5
import time

total_targets = len(target_indexes)  # 目标药物数量
total_trainings = total_targets * n_kfold * k  # 总训练次数
count = 0
global_start = time.time()
for fold in range(n_kfold):
    print("\n\n", fold, "\n\n")
    for train_index, test_index in kfold.split(np.arange(adj_mat_coo_data.shape[0])):
        sampler = Sampler(response_mat=cell_drug, null_mask=null_mask, target_indexes=target_indexes,
                          pos_train_index=train_index, pos_test_index=test_index)
        model = GModel(
            adj_mat=sampler.train_data,
            gene=gene,
            cna=cna,
            mutation=mutation,
            sigma=2,
            k=2,
            iterates=3,
            feature_drug=feature_drug,
            degree=3,
            attention_dim=32,
            n_hid1=192,
            n_hid2=64,
            alpha=8.70,
            device="cuda:0"
        )
        opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                        roc_auc, lr=5e-3, epochs=1000, device="cuda:0").to("cuda:0")
        epoch, true_data, predict_data = opt()
        epochs.append(epoch)
        true_data_s = pd.concat([true_datas, translate_result(true_data)], ignore_index=True)
        predict_data_s = pd.concat([predict_datas, translate_result(predict_data)], ignore_index=True)
        count += 1
        elapsed = time.time() - global_start
        avg_time = elapsed / count
        eta = avg_time * (total_trainings - count)

        print(f"✅ Progress: {count}/{total_trainings} ({(count / total_trainings) * 100:.2f}%) | "
              f"Elapsed: {elapsed / 60:.2f} min | ETA: {eta / 60:.2f} min")

file = open("./result_data/epochs.txt", "w")
file.write(str(epochs))
file.close()
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")



import torch
from CCLE.experiment.myutils import (
    roc_auc, f1_score_binary, accuracy_binary, precision_binary, recall_binary, mcc_binary
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np

true_tensor = (torch.tensor(true_data_s.values.flatten(), dtype=torch.float32) > 0.5).float().cuda()
pred_tensor = torch.tensor(predict_data_s.values.flatten(), dtype=torch.float32).cuda()

n_kfold = 5
fold_size = true_tensor.size(0) // n_kfold

# 分类
cls_metrics = {"AUC": [], "ACC": [], "Precision": [], "Recall": [], "F1": [], "MCC": []}
# 回归
reg_metrics = {"RMSE": [], "MAE": [], "R2": [], "PCC": []}

for i in range(n_kfold):
    start = i * fold_size
    end = (i + 1) * fold_size if i < n_kfold - 1 else true_tensor.size(0)

    t = true_tensor[start:end]
    p = pred_tensor[start:end]

    # NaN 过滤
    mask = ~(torch.isnan(t) | torch.isnan(p))
    t = t[mask]
    p = p[mask]
    t_np = t.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()

    # 分类指标
    auc = roc_auc(t, p)
    f1, threshold = f1_score_binary(t, p)
    acc = accuracy_binary(t, p, threshold)
    precision = precision_binary(t, p, threshold)
    recall = recall_binary(t, p, threshold)
    mcc = mcc_binary(t, p, threshold)

    cls_metrics["AUC"].append(auc)
    cls_metrics["ACC"].append(acc.item())
    cls_metrics["Precision"].append(precision.item())
    cls_metrics["Recall"].append(recall.item())
    cls_metrics["F1"].append(f1.item())
    cls_metrics["MCC"].append(mcc.item())

    # 回归指标
    rmse = np.sqrt(mean_squared_error(t_np, p_np))
    mae = mean_absolute_error(t_np, p_np)
    r2 = r2_score(t_np, p_np)
    pcc = pearsonr(t_np, p_np)[0]

    reg_metrics["RMSE"].append(rmse)
    reg_metrics["MAE"].append(mae)
    reg_metrics["R2"].append(r2)
    reg_metrics["PCC"].append(pcc)


# 打印分类结果
print("\n=== 5-Fold Classification Evaluation Result ===")
for key in cls_metrics:
    values = np.array(cls_metrics[key])
    print(f"{key}: {values.mean():.4f} ± {values.std():.4f}")

# 打印回归结果
print("\n=== 5-Fold Regression Evaluation Result ===")
for key in reg_metrics:
    values = np.array(reg_metrics[key])
    arrow = "↓" if key in ["RMSE", "MAE"] else "↑"
    print(f"{key} {arrow}: {values.mean():.4f} ± {values.std():.4f}")
