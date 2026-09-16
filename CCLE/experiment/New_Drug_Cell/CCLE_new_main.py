
# coding: utf-8
import numpy as np
import pandas as pd
import torch
import os
import time
from CCLE.experiment.myutils import (
    roc_auc, f1_score_binary, accuracy_binary, precision_binary, recall_binary, mcc_binary, translate_result, dir_path
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from CCLE_new_target import fmogc_new_target

data_dir = dir_path(k=2) + "CCLE/processed_data/"
cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
cell_sum = np.sum(cell_drug, axis=1)
drug_sum = np.sum(cell_drug, axis=0)
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)
gene = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)
cna = pd.read_csv(data_dir + "cna_feature.csv", index_col=0, header=0).fillna(0)
cna = np.array(cna, dtype=np.float32)
mutation = pd.read_csv(data_dir + "mutation_feature.csv", index_col=0, header=0)
mutation = np.array(mutation, dtype=np.float32)
mutation = (mutation != 0).astype(np.float32)
null_mask = np.zeros(cell_drug.shape, dtype=np.float32)

n_kfold = 5
os.makedirs("./result_data", exist_ok=True)
true_data_drug = pd.DataFrame()
pred_data_drug = pd.DataFrame()
true_data_cell = pd.DataFrame()
pred_data_cell = pd.DataFrame()

target_dim = [0, 1]
total_targets = sum(np.sum(drug_sum >= 10) if dim == 1 else np.sum(cell_sum >= 10) for dim in target_dim)
total_trainings = total_targets * n_kfold
count = 0
global_start = time.time()

for dim in target_dim:
    for target_index in np.arange(cell_drug.shape[dim]):
        if dim == 0 and cell_sum[target_index] < 10:
            continue
        if dim == 1 and drug_sum[target_index] < 10:
            continue

        for fold in range(n_kfold):
            epoch, true_data, predict_data = fmogc_new_target(
                gene=gene, cna=cna, mutation=mutation, drug_feature=feature_drug,
                response_mat=cell_drug, null_mask=null_mask,
                target_dim=dim, target_index=target_index, evaluate_fun=roc_auc,
                device="cuda:0", degree=2, attention_dim=64
            )
            true_df = translate_result(true_data)
            pred_df = translate_result(predict_data)

            if dim == 0:
                true_data_cell = pd.concat([true_data_cell, true_df], ignore_index=True)
                pred_data_cell = pd.concat([pred_data_cell, pred_df], ignore_index=True)
            else:
                true_data_drug = pd.concat([true_data_drug, true_df], ignore_index=True)
                pred_data_drug = pd.concat([pred_data_drug, pred_df], ignore_index=True)

            count += 1
            elapsed = time.time() - global_start
            avg_time = elapsed / count
            eta = avg_time * (total_trainings - count)
            print(f"✅ Progress: {count}/{total_trainings} "
                  f"({(count / total_trainings) * 100:.2f}%) | "
                  f"Elapsed: {elapsed / 60:.2f} min | ETA: {eta / 60:.2f} min")

def evaluate(true_tensor, pred_tensor, n_kfold=5):
    true_tensor = (torch.tensor(true_tensor.values.flatten(), dtype=torch.float32) > 0.5).float().cuda()
    pred_tensor = torch.tensor(pred_tensor.values.flatten(), dtype=torch.float32).cuda()
    fold_size = true_tensor.size(0) // n_kfold
    cls_metrics = {"AUC": [], "ACC": [], "Precision": [], "Recall": [], "F1": [], "MCC": []}
    reg_metrics = {"RMSE": [], "MAE": [], "R2": [], "PCC": []}
    for i in range(n_kfold):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_kfold - 1 else true_tensor.size(0)
        t = true_tensor[start:end]
        p = pred_tensor[start:end]
        mask = ~(torch.isnan(t) | torch.isnan(p))
        t = t[mask]
        p = p[mask]
        t_np = t.detach().cpu().numpy()
        p_np = p.detach().cpu().numpy()
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
        reg_metrics["RMSE"].append(np.sqrt(mean_squared_error(t_np, p_np)))
        reg_metrics["MAE"].append(mean_absolute_error(t_np, p_np))
        reg_metrics["R2"].append(r2_score(t_np, p_np))
        reg_metrics["PCC"].append(pearsonr(t_np, p_np)[0])
    return cls_metrics, reg_metrics

def print_metrics(name, cls_metrics, reg_metrics):
    print(f"\n=== {name} Classification Evaluation ===")
    for key in cls_metrics:
        values = np.array(cls_metrics[key])
        print(f"{key}: {values.mean():.4f} ± {values.std():.4f}")
    print(f"\n=== {name} Regression Evaluation ===")
    for key in reg_metrics:
        values = np.array(reg_metrics[key])
        arrow = "↓" if key in ["RMSE", "MAE"] else "↑"
        print(f"{key} {arrow}: {values.mean():.4f} ± {values.std():.4f}")

cls_d, reg_d = evaluate(true_data_drug, pred_data_drug)
cls_c, reg_c = evaluate(true_data_cell, pred_data_cell)
print_metrics("New Drug", cls_d, reg_d)
print_metrics("New Cell Line", cls_c, reg_c)
