# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
# from import_path import *
from CCLE.experiment.model import GModel
from CCLE.experiment.optimizer import Optimizer
from sklearn.model_selection import KFold
from sampler import Sampler
from CCLE.experiment.myutils import roc_auc, translate_result, dir_path


data_dir = dir_path(k=2) + "CCLE/processed_data/"

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
adj_mat_coo_data = sp.coo_matrix(cell_drug).data

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)

# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)

# 加载细胞系-cna特征矩阵
cna = np.eye(gene.shape[0], dtype=np.float32)

# 加载细胞系-mutaion特征矩阵
mutation = np.eye(gene.shape[0], dtype=np.float32)

# 加载null_mask
null_mask = np.zeros(cell_drug.shape, dtype=np.float32)

epochs = []
true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=11)

n_kfold = 5
for fold in range(n_kfold):
    for train_index, test_index in kfold.split(np.arange(adj_mat_coo_data.shape[0])):
        sampler = Sampler(cell_drug, train_index, test_index, null_mask)
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
                        roc_auc, lr=1e-3, epochs=1000, device="cuda:0").to("cuda:0")
        epoch, true_data, predict_data = opt()
        epochs.append(epoch)
        true_datas = pd.concat([true_datas, translate_result(true_data)], ignore_index=True)
        predict_datas = pd.concat([predict_datas, translate_result(predict_data)], ignore_index=True)
import os
os.makedirs("./result_data", exist_ok=True)
file = open("./result_data/epochs_gene.txt", "w")
file.write(str(epochs))
file.close()
pd.DataFrame(true_datas).to_csv("./result_data/true_data_gene.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data_gene.csv")



import torch
from CCLE.experiment.myutils import (
    roc_auc, f1_score_binary, accuracy_binary, precision_binary, recall_binary, mcc_binary
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np

true_tensor = (torch.tensor(true_datas.values.flatten(), dtype=torch.float32) > 0.5).float().cuda()
pred_tensor = torch.tensor(predict_datas.values.flatten(), dtype=torch.float32).cuda()

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
