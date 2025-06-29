# analyze_linear_weights.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# === Parameters ===
save_dir = "outputs"           # where to save plots and CSVs
model_tag = "h_d0.0"
model_path = f"models/model_{model_tag}.npy"

# === Load metadata ===
# Make sure these are passed correctly or adapt if you need to import
from scripts.data_loader import extract_metadata, get_protein_coding_gene_indices

gene_symbols, _, _ = extract_metadata("data/raw/human_gene_v2.5.h5")
protein_coding_indices = get_protein_coding_gene_indices("data/raw/human_gene_v2.5.h5")
protein_coding_indices = [i for i in protein_coding_indices if gene_symbols[i] != "IRF2BP2"]

# === Load linear model weights ===
weights = np.load(model_path).flatten()  # includes bias as first element
weights = weights[1:]  # remove bias term

# === Check dimensions ===
assert len(weights) == len(protein_coding_indices), "Mismatch between weights and gene indices"

# === Map weights to gene symbols ===
genes = [gene_symbols[i] for i in protein_coding_indices]
df = pd.DataFrame({"Gene": genes, "Weight": weights})

# === Sort by effect ===
df_sorted = df.sort_values(by="Weight")
df_sorted.reset_index(drop=True, inplace=True)

top_neg = df_sorted.head(30)
top_pos = df_sorted.tail(30)

# === Save CSVs ===
os.makedirs(save_dir, exist_ok=True)
top_neg.to_csv(os.path.join(save_dir, f"top_negative_genes_{model_tag}.csv"), index=False)
top_pos.to_csv(os.path.join(save_dir, f"top_positive_genes_{model_tag}.csv"), index=False)

# === Plot bar chart ===
combined = pd.concat([top_neg, top_pos])
combined_sorted = combined.sort_values(by="Weight")
colors = ["red" if w < 0 else "green" for w in combined_sorted["Weight"]]

plt.figure(figsize=(10, 10))
plt.barh(range(len(combined_sorted)), combined_sorted["Weight"], color=colors)
plt.yticks(range(len(combined_sorted)), combined_sorted["Gene"])
plt.axvline(0, color="black", linestyle="--")
plt.title(f"Top Genes by Linear Model Weight - {model_tag}")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"linear_weights_bar_plot_{model_tag}.png"))
plt.close()
