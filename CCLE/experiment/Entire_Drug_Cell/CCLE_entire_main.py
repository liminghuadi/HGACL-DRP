# coding: utf-8
import numpy as np
import pandas as pd

import torch
torch.autograd.set_detect_anomaly(True)
from CCLE.experiment.model_ours4 import GModel
from CCLE.experiment.optimizer import Optimizer
from sklearn.model_selection import KFold
from sampler import Sampler
from CCLE.experiment.myutils import roc_auc, translate_result, dir_path

import time
import os

data_dir = dir_path(k=2) + "CCLE/processed_data/"

cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0).to_numpy(dtype=np.float32)
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0).to_numpy(dtype=np.float32)
gene = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0).to_numpy(dtype=np.float32)
cna = pd.read_csv(data_dir + "cna_feature.csv", index_col=0, header=0).to_numpy(dtype=np.float32)
mutation = pd.read_csv(data_dir + "mutation_feature.csv", index_col=0, header=0).to_numpy(dtype=np.float32)
mutation = (mutation != 0).astype(np.float32)

null_mask = np.zeros(cell_drug.shape, dtype=np.float32)

k = 5
n_kfold = 1
kfold = KFold(n_splits=k, shuffle=True, random_state=11)

epochs_list = []
all_true_data_folds = []
all_predict_data_folds = []

total_trainings = n_kfold * k
count = 0
global_start = time.time()

sigma = 2
knn = 2
iterates = 3
degree = 3
attention_dim = 32
n_hid1 = 192
n_hid2 = 64
alpha_decoder = 8.70
device = "cuda:0"

learning_rate = 1e-3
max_epochs = 500
test_freq = 20
supervised_weight = 0.7
contrastive_weight = 0.3




for fold_run in range(n_kfold):
    for train_index, test_index in kfold.split(np.arange(cell_drug.shape[0])):

        sampler = Sampler(cell_drug, train_index, test_index, null_mask)

        model = GModel(
            adj_mat=sampler.train_data,
            gene=gene,
            cna=cna,
            mutation=mutation,
            sigma=sigma,
            k=knn,
            iterates=iterates,
            feature_drug=drug_feature,
            degree=degree,
            attention_dim=attention_dim,
            n_hid1=n_hid1,
            n_hid2=n_hid2,
            alpha=alpha_decoder,
            device=device
        )

        opt = Optimizer(
            model,
            sampler.train_data,
            sampler.test_data,
            sampler.test_mask,
            sampler.train_mask,
            roc_auc,
            lr=learning_rate,
            epochs=max_epochs,
            test_freq=test_freq,
            device=device,
            alpha=supervised_weight,
            beta=contrastive_weight
        ).to(device)

        epoch, true_data, predict_data = opt()

        epochs_list.append(epoch)


        count += 1
        elapsed = time.time() - global_start
        avg_time = elapsed / count
        eta = avg_time * (total_trainings - count)

        print(f"✅ Progress: {count}/{total_trainings} "
              f"({(count / total_trainings) * 100:.2f}%) | "
              f"Elapsed: {elapsed / 60:.2f} min | ETA: {eta / 60:.2f} min")

results_for_plotting = []

for fold_idx in range(len(all_true_data_folds)):
    true_df = all_true_data_folds[fold_idx]
    pred_df = all_predict_data_folds[fold_idx]

    true_series = true_df.stack()
    pred_series = pred_df.stack()

    fold_results_df = pd.DataFrame({
        'Fold': fold_idx + 1,
        'True_Label': true_series,
        'Predicted_Score': pred_series
    })

    results_for_plotting.append(fold_results_df)

if results_for_plotting:
    final_plotting_df = pd.concat(results_for_plotting, ignore_index=True)

    output_csv_path = "results_for_roc_pr_curves.csv"
    final_plotting_df.to_csv(output_csv_path, index=False)

    print(f"\n✅ Plotting data successfully saved to: {output_csv_path}")


from CCLE.experiment.myutils import (
    roc_auc, f1_score_binary, accuracy_binary, precision_binary, recall_binary, mcc_binary
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

cls_metrics = {"AUC": [], "ACC": [], "Precision": [], "Recall": [], "F1": [], "MCC": []}
reg_metrics = {"RMSE": [], "MAE": [], "R2": [], "PCC": []}

for i in range(len(all_true_data_folds)):
    true_fold_data = all_true_data_folds[i]
    predict_fold_data = all_predict_data_folds[i]

    true_tensor = (torch.tensor(true_fold_data.values.flatten(), dtype=torch.float32) > 0.5).float().cuda()
    pred_tensor = torch.tensor(predict_fold_data.values.flatten(), dtype=torch.float32).cuda()

    mask = ~(torch.isnan(true_tensor) | torch.isnan(pred_tensor))
    t = true_tensor[mask]
    p = pred_tensor[mask]
    t_np = t.cpu().numpy()
    p_np = p.cpu().numpy()

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

    # Regression Metrics
    rmse = np.sqrt(mean_squared_error(t_np, p_np))
    mae = mean_absolute_error(t_np, p_np)
    r2 = r2_score(t_np, p_np)
    pcc = pearsonr(t_np, p_np)[0]

    reg_metrics["RMSE"].append(rmse)
    reg_metrics["MAE"].append(mae)
    reg_metrics["R2"].append(r2)
    reg_metrics["PCC"].append(pcc)

print("\n=== 5-Fold Cross-Validation Classification Evaluation ===")
for key in cls_metrics:
    values = np.array(cls_metrics[key])
    print(f"{key}: {values.mean():.4f} ± {values.std():.4f}")

print("\n=== 5-Fold Cross-Validation Regression Evaluation ===")
for key in reg_metrics:
    values = np.array(reg_metrics[key])
    arrow = "↓" if key in ["RMSE", "MAE"] else "↑"
    print(f"{key} {arrow}: {values.mean():.4f} ± {values.std():.4f}")
