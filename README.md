# HGACL-DRP

HGACL-DRP predicts cancer-cell-line drug response from cell-line multi-omics data, drug features, and a heterogeneous graph. This README documents the runnable CCLE and GDSC workflows in this repository, including the five-fold main-model runs.

The accompanying [paper](https://doi.org/10.1101/2025.09.22.25336318) is *Heterogeneous Graph Attention Dual-Perturbation Contrastive Learning Network for Drug Response Prediction*. Read [Reproduction scope and known limitations](#reproduction-scope-and-known-limitations) before comparing numbers with the paper.

## 1. Environment

The experiment scripts use a hard-coded `cuda:0` device, so use an NVIDIA GPU with a working CUDA driver. Most commands below use Bash syntax; a PowerShell training example is also provided. Python 3.9 or 3.10 is a practical starting point; the repository does not contain a lockfile or a verified package-version matrix.

Direct Python dependencies are PyTorch, NumPy, pandas, SciPy, scikit-learn, and seaborn. `utils.py` also uses the Python standard library. RMSE is calculated as the square root of `mean_squared_error`, which works with scikit-learn releases that removed the old `squared` argument. One unused fingerprint helper still refers to `np.int`; the main training paths do not call it.

```bash
git clone https://github.com/MingJin426/HGACL-DRP.git
cd HGACL-DRP
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Example for a system that supports the CUDA 11.8 wheel:
python -m pip install 'torch==2.0.1' --index-url https://download.pytorch.org/whl/cu118
# For a different CUDA/driver combination, choose the matching wheel at
# https://pytorch.org/get-started/locally/
python -m pip install 'numpy==1.23.5' 'pandas==1.5.3' 'scipy==1.10.1' \
  'scikit-learn==1.5.2' 'seaborn==0.12.2'
python -c 'import torch, numpy, pandas, scipy, sklearn, seaborn; print(torch.__version__, torch.cuda.is_available())'
```

The last command must print `True` for the supplied training scripts. The package versions above are compatibility suggestions, **not** an environment in which all five-fold experiments have been validated. The [PyTorch installation guide](https://docs.pytorch.org/get-started/previous-versions/) lists the CUDA 11.8 wheel used in the example. The preparation, alignment, and one-fold/one-epoch main-model smoke tests were run on Windows with Python 3.10.20, PyTorch 2.11.0+cu128, NumPy 2.0.1, pandas 2.3.3, SciPy 1.15.3, and scikit-learn 1.7.2. On a Windows console, set `PYTHONIOENCODING=utf-8` before training because progress messages contain Unicode symbols. The shell examples below use Bash syntax; run them in a Bash-compatible shell or translate them for PowerShell.

## 2. Data sources and downloads

**For a repository-level reproduction, use the prepared files included in this repository** ([download the repository](https://github.com/MingJin426/HGACL-DRP)). They include response matrices, drug features, CCLE mutation features, and ZIP-split expression/CNA/GDSC mutation matrices. No separate dataset download is needed after cloning a complete copy.

Original data portals, useful for provenance or rebuilding the dataset, are:

| Data | Official source | Repository input |
| --- | --- | --- |
| CCLE expression, copy number, mutations and response data | [CCLE datasets](https://sites.broadinstitute.org/ccle/datasets), [DepMap downloads](https://depmap.org/portal/data_page/) | `CCLE_processed_data/`, `CCLE_cell_gene_dim_1_num_*.csv.zip`, `CCLE_cell_cna/` |
| GDSC drug response and molecular features | [Sanger GDSC documentation](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/), [Sanger downloads](https://depmap.sanger.ac.uk/downloads/) | `GDSC_processed_data/`, `GDSC_cell_gene_dim_1_num_*.csv.zip`, `GDSC_cell_cna/`, `GDSC_cell_mutation/` |
| Drug structures / PubChem fingerprints | [PubChem downloads](https://pubchem.ncbi.nlm.nih.gov/docs/downloads) | `*/drug_feature.csv` and drug-to-CID mapping CSVs |

Portal releases change over time. Their raw files are **not** interchangeable with the prepared matrices: the repository has no complete raw-to-model preprocessing program or pinned source release. To reproduce the supplied experiments, retain the committed prepared files and their row/column ordering.

The paper describes the label construction as GDSC `IC50 <=` the drug-specific sensitivity threshold and CCLE Z-scored log(IC50) `< -0.8`. The resulting binary labels are already in `GDSC_processed_data/cell_drug_common_binary.csv` and `CCLE_processed_data/cell_drug_binary.csv`. The included GDSC `null_mask.csv` marks unknown associations. Do not treat missing responses as measured resistant samples.

## 3. Prepare the data

Run the following command **from the repository root**. It copies the small prepared CSV files into the locations required by the scripts and joins each family of ZIP archives by columns in numeric shard order. It does not alter the original files. Allow several GB of RAM and free disk space.

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import shutil
import pandas as pd

root = Path.cwd()
for dataset in ('CCLE', 'GDSC'):
    source = root / f'{dataset}_processed_data'
    destination = root / dataset / 'processed_data'
    destination.mkdir(parents=True, exist_ok=True)
    for csv in source.glob('*.csv'):
        shutil.copy2(csv, destination / csv.name)

groups = {
    'CCLE_cell_gene_dim_1_num_*.csv.zip': 'CCLE/processed_data/gene_feature.csv',
    'CCLE_cell_cna/cell_cna_dim_1_num_*.csv.zip': 'CCLE/processed_data/cna_feature.csv',
    'GDSC_cell_gene_dim_1_num_*.csv.zip': 'GDSC/processed_data/cell_gene_feature.csv',
    'GDSC_cell_cna/cell_cna_dim_1_num_*.csv.zip': 'GDSC/processed_data/cell_gene_cna.csv',
    'GDSC_cell_mutation/cell_mutation_dim_1_num_*.csv.zip': 'GDSC/processed_data/cell_gene_mutation.csv',
}
for pattern, output in groups.items():
    parts = sorted(root.glob(pattern), key=lambda p: int(p.name.split('_num_')[-1].split('.')[0]))
    if not parts:
        raise FileNotFoundError(pattern)
    frames = []
    for part in parts:
        with ZipFile(part) as archive, archive.open(archive.namelist()[0]) as stream:
            frame = pd.read_csv(stream, index_col=0)
        if frames and not frame.index.equals(frames[0].index):
            raise ValueError(f'Cell-line order differs: {part}')
        frames.append(frame)
    merged = pd.concat(frames, axis=1)
    if not merged.columns.is_unique:
        raise ValueError(f'Duplicate feature columns: {pattern}')
    target = root / output
    merged.to_csv(target)
    print(f'{target.relative_to(root)}: {merged.shape}')
PY
```

Check row and column alignment before training. Expected response dimensions are **436 × 24** for CCLE and **962 × 228** for GDSC. The model uses array positions, so matching shapes alone are insufficient.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

for name, response, features in (
    ('CCLE', 'cell_drug_binary.csv', ('gene_feature.csv', 'cna_feature.csv', 'mutation_feature.csv')),
    ('GDSC', 'cell_drug_common_binary.csv', ('cell_gene_feature.csv', 'cell_gene_cna.csv', 'cell_gene_mutation.csv')),
):
    directory = Path(name) / 'processed_data'
    labels = pd.read_csv(directory / response, index_col=0)
    drugs = pd.read_csv(directory / 'drug_feature.csv', index_col=0)
    assert labels.columns.astype(str).equals(drugs.index.astype(str))
    for file in features:
        matrix = pd.read_csv(directory / file, index_col=0)
        assert labels.index.astype(str).equals(matrix.index.astype(str)), file
        print(name, file, matrix.shape)
    if name == 'GDSC':
        mask = pd.read_csv(directory / 'null_mask.csv', index_col=0)
        assert labels.index.astype(str).equals(mask.index.astype(str))
        assert labels.columns.astype(str).equals(mask.columns.astype(str))
    print(name, 'response', labels.shape, 'drug features', drugs.shape)
PY
```

For raw-source preprocessing, the paper describes thresholding IC50, standardizing molecular features, mapping drugs to PubChem CIDs, and masking unknown pairs. The exact raw-to-CSV pipeline and source release identifiers are absent here; the supplied prepared matrices are the reproducible input for this checkout.

## 4. Train the main model

Run each entry point from its own experiment directory. This resolves its local `sampler.py` import and keeps output files separate. The main scripts train and evaluate in the same run; no checkpoint or separate prediction command is provided. `set -o pipefail` makes a Python failure visible even when its output is piped through `tee`.

```bash
set -o pipefail
export PYTHONPATH="$PWD"

(cd CCLE/experiment/Entire_Drug_Cell && python CCLE_entire_main_ours4.py 2>&1 | tee run_full.log)
(cd GDSC/experiment/Entire_Drug_Cell && python GDSC_entire_main.py 2>&1 | tee run_full.log)
```

Both full-model scripts use five folds (`KFold`, shuffle enabled, `random_state=11`), 500 epochs, learning rate `1e-3`, and `cuda:0`. CCLE uses supervised/contrastive weights `0.7/0.3`; GDSC uses `0.9/0.1`. The exact model settings are near the beginning of each script. Negative samples are drawn without an explicit NumPy seed, so repeated runs can differ.

On Windows PowerShell, after activating a compatible Conda environment and completing Section 3, run the same main experiments from the repository root as follows:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONIOENCODING = 'utf-8'
Push-Location CCLE/experiment/Entire_Drug_Cell
python CCLE_entire_main_ours4.py 2>&1 | Tee-Object run_full.log
if ($LASTEXITCODE -ne 0) { throw 'CCLE training failed' }
Pop-Location
Push-Location GDSC/experiment/Entire_Drug_Cell
python GDSC_entire_main.py 2>&1 | Tee-Object run_full.log
if ($LASTEXITCODE -ne 0) { throw 'GDSC training failed' }
Pop-Location
```

For the Python heredoc blocks in Sections 3 and 5, PowerShell users can save the code between `python - <<'PY'` and `PY` as a `.py` file and run `python path/to/file.py` from the repository root.

The main outputs are:

| Run | Log and predictions |
| --- | --- |
| CCLE | `CCLE/experiment/Entire_Drug_Cell/run_full.log`; `results_for_roc_curves.csv` in the same directory |
| GDSC | `GDSC/experiment/Entire_Drug_Cell/run_full.log`; `results_for_roc_curves_oursGDSC.csv` in the same directory |

Each prediction CSV contains `Fold`, `True_Label`, and `Predicted_Score`. The log prints the script's per-fold and mean ± standard-deviation metrics. `CCLE_entire_main.py` is a separate variant whose fold-result lists are not populated; use `CCLE_entire_main_ours4.py` for the CCLE full-model run.

## 5. Evaluate saved predictions

The following **standalone evaluation command** reads the two prediction CSVs after the main runs. It prints fold-level and mean ± standard-deviation ROC-AUC, PR-AUC, ACC, Precision, Recall, F1, and MCC. The classification threshold is chosen separately for each fold to maximize F1, as in the provided evaluation code. PR-AUC is the trapezoidal area under the precision–recall curve; the training scripts themselves do not print it.

```bash
python - <<'PY'
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

files = {
    'CCLE': 'CCLE/experiment/Entire_Drug_Cell/results_for_roc_curves.csv',
    'GDSC': 'GDSC/experiment/Entire_Drug_Cell/results_for_roc_curves_oursGDSC.csv',
}
for dataset, filename in files.items():
    data = pd.read_csv(filename).dropna(subset=['Fold', 'True_Label', 'Predicted_Score'])
    rows = []
    for fold, group in data.groupby('Fold', sort=True):
        y = (group['True_Label'].to_numpy(dtype=float) > 0.5).astype(int)
        score = group['Predicted_Score'].to_numpy(dtype=float)
        if len(np.unique(y)) != 2:
            raise ValueError(f'{dataset} fold {fold} has only one class')
        precision, recall, thresholds = precision_recall_curve(y, score)
        f1 = np.divide(2 * precision[:-1] * recall[:-1],
                       precision[:-1] + recall[:-1],
                       out=np.zeros_like(precision[:-1]),
                       where=(precision[:-1] + recall[:-1]) != 0)
        threshold = thresholds[np.argmax(f1)]
        predicted = (score >= threshold).astype(int)
        values = {
            'AUC': roc_auc_score(y, score),
            'AUPR': auc(recall, precision),
            'ACC': accuracy_score(y, predicted),
            'Precision': precision_score(y, predicted, zero_division=0),
            'Recall': recall_score(y, predicted, zero_division=0),
            'F1': f1.max(),
            'MCC': matthews_corrcoef(y, predicted),
        }
        rows.append(values)
        print(dataset, 'fold', fold, values)
    result = pd.DataFrame(rows)
    print(dataset, 'mean ± std')
    for metric in result:
        print(f'  {metric}: {result[metric].mean():.4f} ± {result[metric].std(ddof=0):.4f}')
PY
```

The script's final log also prints RMSE, MAE, R² and PCC. Its optimizer selects the best epoch by **test-fold AUC**; the displayed score is therefore not an independently selected held-out estimate. Preserve the exact CSV and log from each run when comparing results.

## 6. Reproduce the other available experiments

The following CCLE entry points are present. Run them after Section 3, one at a time; each trains and evaluates in one process. Per-drug runs can require substantially more time than the five-fold main run.

```bash
set -o pipefail
export PYTHONPATH="$PWD"

# Gene-expression-only comparison
(cd CCLE/experiment/Entire_Drug_Cell && python entire_gene_main.py 2>&1 | tee run_gene.log)

# Hard-coded target drug CID 16038120
(cd CCLE/experiment/Target_Drug && python CCLE_target_main.py 2>&1 | tee run_target.log)

# Each drug with at least 10 positive pairs; five repeats of five folds
(cd CCLE/experiment/Single_Drug && python CCLE_single_main.py 2>&1 | tee run_single.log)

# Held-out cell-line and held-out drug scenarios
(cd CCLE/experiment/New_Drug_Cell && python CCLE_new_main.py 2>&1 | tee run_new.log)
```

The target-drug script writes `result_data/{epochs.txt,true_data.csv,predict_data.csv}` relative to its experiment directory. The single-drug script writes one true/prediction CSV per drug under `pan_result_data/`. Re-running in the same directory can overwrite outputs; save each run's directory or logs before changing settings.

GDSC `Target_Drug`, `Single_Drug`, and `New_Drug_Cell` entry points exist, but they import `GDSC.experiment.model`, `GDSC.experiment.optimizer`, and/or `GDSC.experiment.myutils`, which are **missing from this repository**. After the authors provide those modules, their commands are:

```bash
set -o pipefail
export PYTHONPATH="$PWD"
(cd GDSC/experiment/Target_Drug && python GDSC_target_main.py 2>&1 | tee run_target.log)
(cd GDSC/experiment/Single_Drug && mkdir -p pan_result_data && python GDSC_single_main.py 2>&1 | tee run_single.log)
(cd GDSC/experiment/New_Drug_Cell && python GDSC_new_main.py 2>&1 | tee run_new.log)
```

The `mkdir` is necessary because `GDSC_single_main.py` opens a file there before creating the directory. Do not silently substitute CCLE modules for the missing GDSC modules; their settings may differ.

## Reproduction scope and known limitations

| Paper result | Reproduction path in this checkout |
| --- | --- |
| Main HGACL-DRP CCLE and GDSC classification rows in Tables 1–2 | Section 4 training and Section 5 evaluation; compare fold means with the [paper](https://doi.org/10.1101/2025.09.22.25336318). |
| ROC and precision–recall curves for HGACL-DRP | Use the main-run prediction CSVs from Section 4; no plotting script is included. |
| Gene-only CCLE, single-drug, target-drug, and held-out cell/drug results | CCLE commands in Section 6; GDSC commands require missing modules. |
| Seven baseline methods and full ablation comparison | Baseline/ablation implementations and experiment configurations are absent, so these paper results cannot be reproduced from this checkout alone. |

The [paper](https://doi.org/10.1101/2025.09.22.25336318) reports HGACL-DRP AUC `0.9548` on CCLE and `0.9899` on GDSC. These are reference values from the paper, **not verified outputs of this checkout**. The repository has no saved model checkpoints, pinned dependency lockfile, or complete original-data preprocessing pipeline. These gaps, the test-fold epoch selection, and random negative sampling limit exact numerical reproduction.

Verification to date covered the complete data preparation/alignment commands, import and syntax checks, one real-data fold with one training epoch for each main model, and the standalone evaluation command on those smoke-test predictions. It did **not** cover the full five-fold, 500-epoch runs or missing baseline/ablation experiments.

## Citation

Min Li, Ming Jin, and Shaobo Deng. *HGACL-DRP: Heterogeneous Graph Attention Dual-Perturbation Contrastive Learning Network for Drug Response Prediction.* [medRxiv, 2025](https://doi.org/10.1101/2025.09.22.25336318).
