import os
import pandas as pd
from glob import glob

def summarise_top_gene_matrix(output_folder="outputs/promising_results/top_genes", 
                              attribution_folder="outputs/attributions_top_genes", 
                              output_file="top_gene_matrix.csv"):
    """
    Summarises top genes from multiple models (positive and negative), showing:
    - Which models each gene appears in
    - Total counts across models
    - Average attribution score (from full attribution files)
    Saves a unified matrix to the specified output file.
    """

    def collect_gene_appearances(file_pattern, suffix):
        files = glob(os.path.join(output_folder, file_pattern))
        gene_to_models = {}

        for filepath in files:
            filename = os.path.basename(filepath)
            model_tag = filename.replace(f"top_{suffix}_genes_", "").replace(".csv", "") + f"_{suffix}"

            df = pd.read_csv(filepath)
            genes = df["Gene"].tolist()

            for gene in genes:
                if gene not in gene_to_models:
                    gene_to_models[gene] = {}
                gene_to_models[gene][model_tag] = 1

        return gene_to_models

    def collect_attribution_scores():
        files = glob(os.path.join(attribution_folder, "all_gene_attributions_*.csv"))
        gene_scores = {}

        for filepath in files:
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                gene = row["Gene"]
                score = row["Attribution_Score"]
                if gene not in gene_scores:
                    gene_scores[gene] = []
                gene_scores[gene].append(score)

        # Compute mean attribution per gene
        return {gene: sum(scores)/len(scores) for gene, scores in gene_scores.items()}

    # === Collect top gene presence ===
    neg_dict = collect_gene_appearances("top_negative_genes_*.csv", "neg")
    pos_dict = collect_gene_appearances("top_positive_genes_*.csv", "pos")

    all_genes = sorted(set(neg_dict) | set(pos_dict))
    all_configs = sorted({tag for d in (neg_dict, pos_dict) for gene in d for tag in d[gene]})

    matrix = pd.DataFrame(0, index=all_genes, columns=all_configs)

    for gene_dict in [neg_dict, pos_dict]:
        for gene, tag_map in gene_dict.items():
            for tag in tag_map:
                matrix.at[gene, tag] = 1

    matrix["Total_Neg"] = matrix[[c for c in matrix.columns if c.endswith("_neg")]].sum(axis=1)
    matrix["Total_Pos"] = matrix[[c for c in matrix.columns if c.endswith("_pos")]].sum(axis=1)

    # === Add average attribution score ===
    avg_attr_scores = collect_attribution_scores()
    matrix["Avg_Attribution"] = matrix.index.map(lambda g: avg_attr_scores.get(g, 0))

    # === Save ===
    output_path = os.path.join(output_folder, output_file)
    matrix.to_csv(output_path)
    print(f"\nUnified top gene matrix saved to: {output_path}")

# execute the function
summarise_top_gene_matrix()