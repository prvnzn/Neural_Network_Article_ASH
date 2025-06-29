# scripts/summarised_overlap.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from collections import Counter

# === Parameters ===
linear_dir = "outputs/linear_model/proper_run"
nn_dir = "outputs/promising_results/top_genes"
linear_tag = "h_d0.0"
min_model_threshold = 2
output_csv = os.path.join(linear_dir, "summarised_overlap_genes_filtered.csv")

# === Load linear model genes ===
linear_pos = pd.read_csv(os.path.join(linear_dir, f"top_positive_genes_{linear_tag}.csv"))["Gene"]
linear_neg = pd.read_csv(os.path.join(linear_dir, f"top_negative_genes_{linear_tag}.csv"))["Gene"]
linear_pos_set = set(linear_pos)
linear_neg_set = set(linear_neg)

# === Scan all NN models ===
nn_pos_counter = Counter()
nn_neg_counter = Counter()

for file in os.listdir(nn_dir):
    if file.startswith("top_positive_genes_"):
        df = pd.read_csv(os.path.join(nn_dir, file))
        nn_pos_counter.update(df["Gene"])
    elif file.startswith("top_negative_genes_"):
        df = pd.read_csv(os.path.join(nn_dir, file))
        nn_neg_counter.update(df["Gene"])

# === Filter by frequency (≥2 NN models) ===
nn_pos_filtered = {gene for gene, count in nn_pos_counter.items() if count >= min_model_threshold}
nn_neg_filtered = {gene for gene, count in nn_neg_counter.items() if count >= min_model_threshold}

# === Compare to linear model ===
pos_overlap = sorted(nn_pos_filtered & linear_pos_set)
neg_overlap = sorted(nn_neg_filtered & linear_neg_set)

# === Save overlap gene list with counts ===
df_pos = pd.DataFrame({
    "Gene": pos_overlap,
    "NN_Model_Count": [nn_pos_counter[g] for g in pos_overlap]
})
df_neg = pd.DataFrame({
    "Gene": neg_overlap,
    "NN_Model_Count": [nn_neg_counter[g] for g in neg_overlap]
})

# Merge for output
max_len = max(len(df_pos), len(df_neg))
df_pos = df_pos.reindex(range(max_len))
df_neg = df_neg.reindex(range(max_len))

df_out = pd.DataFrame({
    "Positive_Overlap_Genes": df_pos["Gene"],
    "Positive_NN_Model_Count": df_pos["NN_Model_Count"],
    "Negative_Overlap_Genes": df_neg["Gene"],
    "Negative_NN_Model_Count": df_neg["NN_Model_Count"]
})
df_out.to_csv(output_csv, index=False)

# === Plot final labelled Venns ===
def plot_venn(set_nn, set_linear, overlap_genes, title, out_name):
    venn = venn2([set_nn, set_linear], set_labels=("NN Models (≥2)", "Linear Model"))
    if venn.get_label_by_id("10"):
        venn.get_label_by_id("10").set_text("")
    if venn.get_label_by_id("01"):
        venn.get_label_by_id("01").set_text("")
        venn.get_label_by_id("11").set_text("\n".join(overlap_genes))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(linear_dir, out_name))
    plt.close()

# Plot positive
plot_venn(
    set_nn=nn_pos_filtered,
    set_linear=linear_pos_set,
    overlap_genes=pos_overlap,
    title="Summarised Venn – Positive Regulators (≥2 NN Models)",
    out_name="summarised_venn_positive_filtered.png"
)

# Plot negative
plot_venn(
    set_nn=nn_neg_filtered,
    set_linear=linear_neg_set,
    overlap_genes=neg_overlap,
    title="Summarised Venn – Negative Regulators (≥2 NN Models)",
    out_name="summarised_venn_negative_filtered.png"
)
