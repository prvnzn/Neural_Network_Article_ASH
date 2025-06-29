
# IRF2BP2 Neural Network Attribution Pipeline: Project Summary

This document summarises the components of a machine learning pipeline aimed at identifying gene-level drivers of **IRF2BP2 expression** using bulk RNA-seq data and neural network interpretability techniques. A closed-form linear model was also explored as a complementary baseline.

---

## 1. Data Source & Preprocessing

### Data:

* **Source**: ARCHS4 bulk RNA-seq `.h5` dataset
* **Content**: TPM-normalised gene expression matrix, gene metadata, and sample annotations

### Filtering Steps:

* **Excluded single-cell samples**: `singlecellprobability > 0.5`
* **Selected myeloid-lineage samples** using metadata keyword matching (e.g. `"monocyte"`, `"AML"`, `"U937"`, etc.)
* **Retained only protein-coding genes**
* **Target variable**: IRF2BP2 expression (removed from input matrix and set as `y`)

### Output:

* Deterministic hash-based train/val/test split using sample IDs
* Saved as:

  * `data/processed/train_data_log1p.npz`
  * `data/processed/val_data_log1p.npz`
  * `data/processed/test_data_log1p.npz`

---

## 2. Model Architecture & Training

### Neural Network Models:

* Implemented in **PyTorch**
* Feedforward Multilayer Perceptrons (MLPs) with various depths and dropout settings
* Trained using **log1p-transformed data**, **MSE loss**, and **Adam** optimiser

Example configuration:
```python
model_configs = [
    {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [128, 64], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [32], "dropout": 0.1}
]
```

### Output:

* Best models saved as `.pth`
* Training loss plots and prediction scatterplots generated per model

---

## 3. Feature Attribution

### Approach:

* Applied **Gradient × Input** attribution to test samples
* For each trained NN model:
  - Extracted **top 30 positive and negative genes**
  - Saved as:
    * `top_positive_genes_<tag>.csv`
    * `top_negative_genes_<tag>.csv`
  - Generated directional attribution bar plots

---

## 4. Linear Model Baseline

* A **closed-form linear regression model** was implemented as a baseline
* No training required — fitted using pseudoinverse
* Weights were extracted and interpreted as gene coefficients
* Top 30 positive and negative genes by weight magnitude were plotted
* Output stored under: `outputs/linear_model/proper_run/`

This model served as a **complementary validation step** to the neural network pipeline, offering a simple yet interpretable comparison.

---

## 5. Aggregated Gene Summary Across Models

To identify **robust regulators**, gene appearances across all NN models were aggregated:

* Built: `top_gene_matrix.csv`
* Computed for each gene:
  - `Total_Pos`, `Total_Neg`
  - `Total_Models` = sum of appearances
  - `Direction` = net polarity (positive or negative)
  - `Stability_Score` = # of models in which the gene appears
  - `Directional_Confidence` = |Pos - Neg|
  - `Confidence_Score` = Stability_Score × Directional_Confidence

---

## 6. Final Visualisations & Ranked Summaries

* Genes were re-ranked based on **attribution score**, not just frequency
* Two dedicated plots were created:
  - `top30_positive_regulators.png`
  - `top30_negative_regulators.png`
* Gene labels include model count: e.g., `JUN (6)`
* Full table stored as: `top30_positive_and_negative_regulators.csv`

---

## 7. Linear–NN Overlap Analysis

* All NN model outputs were pooled and compared to the linear model gene lists
* Created summary overlap plots:
  - `summarised_venn_positive_filtered.png`
  - `summarised_venn_negative_filtered.png`
* Overlapping genes (appearing in linear + ≥2 NN models) saved as:
  - `summarised_overlap_genes_filtered.csv`

---

## Final Output Artifacts

```
data/processed/train_data_log1p.npz
data/processed/val_data_log1p.npz
data/processed/test_data_log1p.npz

models/model_*.pth

outputs/promising_results/top_genes/top_positive_genes_*.csv
outputs/promising_results/top_genes/top_negative_genes_*.csv

outputs/top_gene_matrix.csv
outputs/top30_positive_and_negative_regulators.csv
outputs/top30_positive_regulators.png
outputs/top30_negative_regulators.png

outputs/linear_model/proper_run/top_positive_genes_h_d0.0.csv
outputs/linear_model/proper_run/top_negative_genes_h_d0.0.csv
outputs/linear_model/proper_run/nn_linear_overlap_venn.png
outputs/linear_model/proper_run/summarised_overlap_genes_filtered.csv
outputs/linear_model/proper_run/summarised_venn_positive_filtered.png
outputs/linear_model/proper_run/summarised_venn_negative_filtered.png
```

---

## Conclusion

This pipeline identifies **robust candidate regulators** of IRF2BP2 in AML-relevant myeloid transcriptomes. The inclusion of a linear model provided a useful benchmark, confirming several overlapping hits with our neural network models. The most stable and directionally consistent candidates can be prioritised for **experimental validation**.
