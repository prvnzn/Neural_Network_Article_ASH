# scripts/NN_linear_link.py

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# === Fixed Paths ===
linear_dir = "outputs/linear_model/proper_run"
nn_dir = "outputs/promising_results/top_genes"
output_dir = linear_dir  # all comparison plots go here

linear_tag = "h_d0.0"
linear_pos_path = os.path.join(linear_dir, f"top_positive_genes_{linear_tag}.csv")
linear_neg_path = os.path.join(linear_dir, f"top_negative_genes_{linear_tag}.csv")

# === Load Linear Data ===
linear_pos = pd.read_csv(linear_pos_path)
linear_neg = pd.read_csv(linear_neg_path)
linear_pos_genes = set(linear_pos["Gene"])
linear_neg_genes = set(linear_neg["Gene"])
df_linear_all = pd.concat([linear_pos, linear_neg], axis=0)

# === Loop over NN result files ===
all_files = os.listdir(nn_dir)
nn_tags = sorted(set(f.replace("top_positive_genes_", "").replace("top_negative_genes_", "").replace(".csv", "")
                    for f in all_files if f.startswith("top_")))

for nn_tag in nn_tags:
    nn_pos_path = os.path.join(nn_dir, f"top_positive_genes_{nn_tag}.csv")
    nn_neg_path = os.path.join(nn_dir, f"top_negative_genes_{nn_tag}.csv")

    if not (os.path.exists(nn_pos_path) and os.path.exists(nn_neg_path)):
        print(f"Skipping incomplete NN model outputs for tag: {nn_tag}")
        continue

    nn_pos = pd.read_csv(nn_pos_path)
    nn_neg = pd.read_csv(nn_neg_path)
    nn_pos_genes = set(nn_pos["Gene"])
    nn_neg_genes = set(nn_neg["Gene"])
    df_nn_all = pd.concat([nn_pos, nn_neg], axis=0)

    # === Compute overlaps ===
    pos_overlap = linear_pos_genes & nn_pos_genes
    neg_overlap = linear_neg_genes & nn_neg_genes

    print(f"[{nn_tag}] Positive Overlap ({len(pos_overlap)}): {sorted(pos_overlap)}")
    print(f"[{nn_tag}] Negative Overlap ({len(neg_overlap)}): {sorted(neg_overlap)}")

    # === Save Venn Diagram ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    venn2([linear_pos_genes, nn_pos_genes], set_labels=("Linear Pos", f"NN Pos ({nn_tag})"))
    plt.title("Top Positive Overlap")

    plt.subplot(1, 2, 2)
    venn2([linear_neg_genes, nn_neg_genes], set_labels=("Linear Neg", f"NN Neg ({nn_tag})"))
    plt.title("Top Negative Overlap")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"venn_overlap_{nn_tag}.png"))
    plt.close()

    # === Save Scatter Plot ===
    merged = pd.merge(df_linear_all, df_nn_all, on="Gene", suffixes=("_Linear", "_NN"))

    merged_pos = merged[merged["Gene"].isin(pos_overlap)]
    merged_neg = merged[merged["Gene"].isin(neg_overlap)]

    plt.figure(figsize=(8, 6))
    if not merged_pos.empty:
        plt.scatter(merged_pos["Weight"], merged_pos["Attribution_Score"], color="green", label="Positive Overlap")
    if not merged_neg.empty:
        plt.scatter(merged_neg["Weight"], merged_neg["Attribution_Score"], color="red", label="Negative Overlap")

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.xlabel("Linear Model Weight")
    plt.ylabel("NN Attribution Score")
    plt.title(f"Gene Overlap: NN vs Linear ({nn_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_overlap_{nn_tag}.png"))
    plt.close()
