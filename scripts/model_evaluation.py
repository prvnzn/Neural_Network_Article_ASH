import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mygene import MyGeneInfo
from matplotlib.patches import Patch

# === Load top gene matrix ===
df = pd.read_csv("outputs/promising_results/top_genes/top_gene_matrix.csv", index_col=0)

# === Compute composite scores ===
df["Total_Models"] = df["Total_Neg"] + df["Total_Pos"]
df["Direction"] = np.where(df["Total_Pos"] > df["Total_Neg"], "positive", "negative")
df["Stability_Score"] = df["Total_Models"]
df["Directional_Confidence"] = np.abs(df["Total_Pos"] - df["Total_Neg"])
df["Confidence_Score"] = df["Stability_Score"] * df["Directional_Confidence"]
df["Avg_Attribution_Abs"] = df["Avg_Attribution"].abs()

# === Split and rank separately
df_pos = df[df["Total_Pos"] > df["Total_Neg"]].copy()
df_neg = df[df["Total_Neg"] > df["Total_Pos"]].copy()

df_pos = df_pos.sort_values(by="Avg_Attribution", ascending=False).head(30)
df_neg = df_neg.sort_values(by="Avg_Attribution", ascending=True).head(30)

# === Map Ensembl to symbol
def is_ensembl_id(x): return str(x).startswith("ENSG")
ensembl_ids = [i for i in list(df_pos.index) + list(df_neg.index) if is_ensembl_id(i)]

symbol_map = {}
if ensembl_ids:
    mg = MyGeneInfo()
    query_results = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human")
    for entry in query_results:
        if not entry.get("notfound", False):
            symbol_map[entry["query"]] = entry.get("symbol", entry["query"])

# === Apply gene symbol + model count labels
for df_sub in [df_pos, df_neg]:
    df_sub["symbol"] = [symbol_map.get(g, g) for g in df_sub.index]
    df_sub["symbol_plot"] = df_sub.apply(lambda row: f"{row['symbol']} ({row['Total_Models']})", axis=1)

# === Plot Positive Regulators ===
plt.figure(figsize=(10, 8))
plt.barh(df_pos["symbol_plot"], df_pos["Confidence_Score"], color="lightgreen")
plt.xlabel("Model Confidence Score")
plt.title("Top 30 Positive Regulators of IRF2BP2")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/top30_positive_regulators.png")
plt.close()

# === Plot Negative Regulators ===
plt.figure(figsize=(10, 8))
plt.barh(df_neg["symbol_plot"], df_neg["Confidence_Score"], color="lightcoral")
plt.xlabel("Model Confidence Score")
plt.title("Top 30 Negative Regulators of IRF2BP2")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/top30_negative_regulators.png")
plt.close()

# === Combine and Save CSV
df_combined = pd.concat([df_pos, df_neg])
df_combined.to_csv("outputs/top30_positive_and_negative_regulators.csv")
print("Saved: outputs/top30_positive_and_negative_regulators.csv")
