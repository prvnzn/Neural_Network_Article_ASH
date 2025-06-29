# .h5 file containing all the genes and the samples

import h5py
import os
import numpy as np

def explore_h5_structure(filepath):
    """
    Opens the .h5 file and prints its internal structure.
    Helps us understand how to access gene expression data and metadata.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with h5py.File(filepath, 'r') as f:
        print("Top-level keys in the H5 file:")
        for key in f.keys():
            print(f" - {key}")

        print("\nInspecting 'meta' group:")
        if 'meta' in f:
            for subkey in f['meta'].keys():
                print(f" - meta/{subkey}")

        print("\nInspecting 'data' group:")
        if 'data' in f:
            for subkey in f['data'].keys():
                print(f" - data/{subkey}")

        print("\nExpression matrix shape:")
        expression_data = f['data']['expression']
        print(f" - shape: {expression_data.shape}")

        # Decode gene and sample names properly
        print("\nContents of meta/genes:")
        for subkey in f['meta']['genes'].keys():
            print(f" - meta/genes/{subkey}")

        print("\nContents of meta/samples:")
        for subkey in f['meta']['samples'].keys():
            print(f" - meta/samples/{subkey}")


def extract_metadata(filepath):
    """
    Extracts and decodes useful metadata from the .h5 file:
    - Gene symbols
    - Sample source descriptions (for filtering)
    - Single-cell probability scores (for filtering)
    """
    with h5py.File(filepath, 'r') as f:
        # Extract gene symbols
        gene_symbols = [g.decode('utf-8') for g in f['meta']['genes']['symbol'][:]]

        # Extract sample descriptions
        source_names = [s.decode('utf-8') for s in f['meta']['samples']['source_name_ch1'][:]]

        # Extract single-cell probability
        singlecell_prob = f['meta']['samples']['singlecellprobability'][:]
        singlecell_prob = np.array(singlecell_prob)  # Convert to NumPy for easy filtering

        print(f"\nTotal genes: {len(gene_symbols)}")
        print(f"Total samples: {len(source_names)}")
        print(f"Single-cell entries with prob > 0.5: {np.sum(singlecell_prob > 0.5)}")

        return gene_symbols, source_names, singlecell_prob
    

def filter_myeloid_bulk_samples(source_names, singlecell_prob):
    """
    Filters source names to find myeloid lineage samples that are bulk RNA-seq.
    Returns the indices of samples to keep.
    """
    import numpy as np

    # Define myeloid-related keywords (expand as needed)
    myeloid_keywords = [
        "monocyte", "macrophage", "myeloid", "dendritic", "AML", "CD14",
        "MDS", "U937", "THP", "HL-60", "MOLM", "NB4", "MV4", "MUTZ", "CD33"
    ]

    # Lowercase everything for safer matching
    source_names = [s.lower() for s in source_names]

    # Build filter mask
    valid_indices = []

    for i, (desc, prob) in enumerate(zip(source_names, singlecell_prob)):
        if prob <= 0.5:  # Must be bulk
            for keyword in myeloid_keywords:
                if keyword in desc:
                    valid_indices.append(i)
                    break  # Stop checking keywords once one matches

    print(f"\nTotal valid (bulk + myeloid) samples: {len(valid_indices)}")
    return valid_indices


def get_protein_coding_gene_indices(filepath):
    """
    Returns indices of genes that are labeled as 'protein_coding'
    based on the meta/genes/biotype field.
    """
    with h5py.File(filepath, 'r') as f:
        # Load gene biotypes
        biotypes = [b.decode('utf-8') for b in f['meta']['genes']['biotype'][:]]
        
        # Filter indices
        protein_coding_indices = [i for i, bt in enumerate(biotypes) if bt == "protein_coding"]
        
        print(f"Found {len(protein_coding_indices)} protein-coding genes.")
        return protein_coding_indices


def extract_and_save_expression_matrix(filepath, gene_symbols, sample_metadata, gene_row_indices, target_gene, save_filename):
    """
    Extracts and saves the gene expression matrix (X) and target values (y)
    for a given subset of samples (train, val, test).
    """
    import numpy as np
    import os
    import h5py

    save_path = os.path.join("data", "processed", save_filename)

    print(f"\n Extracting: {save_filename}", flush=True)

    with h5py.File(filepath, 'r') as f:
        expression = f['data']['expression']

        # Get index of IRF2BP2
        try:
            irf2bp2_index = gene_symbols.index(target_gene)
        except ValueError:
            raise ValueError(f"{target_gene} not found in gene list!")

        print(f"\nExtracting for {save_filename}:")
        print(f"{target_gene} is at row index {irf2bp2_index}.")

        # Remove IRF2BP2 from feature list
        gene_row_indices = [idx for idx in gene_row_indices if idx != irf2bp2_index]

        valid_sample_indices = np.array(sorted(sample_metadata["sample_index"].tolist()), dtype=np.int64)



        print(f"Building matrix with {len(gene_row_indices)} genes and {len(valid_sample_indices)} samples...")

        # Preallocate final X
        X = np.zeros((len(valid_sample_indices), len(gene_row_indices)), dtype=np.float32)

        # Fill X
        for j, gene_idx in enumerate(gene_row_indices):
            X[:, j] = np.array(expression[gene_idx, valid_sample_indices], dtype=np.float32)


            if j % 1000 == 0 or j == len(gene_row_indices) - 1:
                print(f"Processed gene {j + 1}/{len(gene_row_indices)}")

        # Extract target y
        y = expression[irf2bp2_index, valid_sample_indices]
        y = np.array(y, dtype=np.float32)

        # Save immediately
        np.savez_compressed(save_path, X=X, y=y)
        print(f"Saved extracted data to {save_path}")





# accessing the processed data
'''
How to access the saved files: data = np.load("data/processed/myeloid_bulk.npz")
X, y = data["X"], data["y"]
'''


import pandas as pd

def build_sample_metadata(source_names, valid_sample_indices):
    """
    Builds a DataFrame containing sample index, sample ID, and description.
    Used for stable hash-based splitting.
    """
    # Create a simple DataFrame
    metadata = pd.DataFrame({
        "sample_index": valid_sample_indices,
        "sample_id": [i for i in valid_sample_indices],  # use integer indices as unique IDs
        "description": [source_names[i] for i in valid_sample_indices]
    })

    print(f"Built metadata table with {len(metadata)} samples")
    return metadata


from zlib import crc32
import pandas as pd

def is_id_in_test_set(identifier, test_ratio=0.2):
    """
    Returns True if the hash of the identifier places it in the test set.
    """
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32 # hash function

def split_train_test(metadata, test_ratio=0.2, id_column="sample_id"):
    """
    Splits metadata into training and test set based on stable hashing.
    """
    ids = metadata[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return metadata.loc[~in_test_set], metadata.loc[in_test_set]

def split_train_val(train_metadata, val_ratio=0.15):
    """
    Random split of training metadata into train and validation sets.
    """
    shuffled = train_metadata.sample(frac=1, random_state=42)  # shuffle
    n_val = int(len(shuffled) * val_ratio)
    val_set = shuffled.iloc[:n_val]
    train_set = shuffled.iloc[n_val:]
    return train_set, val_set



### Creating the datasets from the .h5 files: 

# Exploring the .h5 file containing the samples (rows) and genes (columns) with their corresponding TPM 
from scripts.data_loader import explore_h5_structure

# Provides an overview on what the data and meta data folders contain in the .h5 file
''' 
explore_h5_structure("data/raw/human_gene_v2.5.h5")
'''
# .h5 data structure explanation:
'''
when opening the .h5 file, we have sub-folders
data: holds the actual gene expression values
meta: holds all the labels that tell you what the expression values mean (e.g. sample names, gene names, etc.)
within the sub-folders, there are additional subfolders that can store data:
Inspecting 'meta' group:
 - meta/genes (A list of all gene names)
 - meta/info (Tissue, cell type, experimental setup, disease)
 - meta/samples (A list of all sample identifiers)

Inspecting 'data' group:
 - data/expression (Very likely shape: [n_genes, n_samples])
'''

# data/expression matrix format: genes (rows) X samples (columns)

# Info on the approach
'''
# before accessing the expression matrix, we will create lists with appropriate metadata allowing us to isolate only
# the samples that are relevant for our neural network

# accessing the genes (rows), the samples (columns) and determine which samples are likely scRNA-seq data
# we only want to keep the samples which are very likely bulk RNA-seq data and not scRNA-seq
# as we train our model on bulk RNA-seq
# the variable singlecell_prob provides us the amount of samples that are very likely scRNA-seq data meaning we need
# to exclude those samples
'''

# samples with single-cell prob >0.5 are very likely to be scRNA-seq (needs to be removed)
# source names provide sample description
from scripts.data_loader import extract_metadata
gene_symbols, source_names, singlecell_prob = extract_metadata("data/raw/human_gene_v2.5.h5")

### Extract only relevant columns (AML-specific samples)
from scripts.data_loader import filter_myeloid_bulk_samples
# Filtering the data and only keeping the myeloid cell lineages that were bulk RNA-sequenced
valid_indices = filter_myeloid_bulk_samples(source_names, singlecell_prob)

# Building metadata table to index the valid samples    
from scripts.data_loader import build_sample_metadata
metadata = build_sample_metadata(source_names, valid_indices)

### Splitting the complete expression matrix in train, validation and test data sets
from scripts.data_loader import split_train_test, split_train_val
# First split into train and test
train_metadata, test_metadata = split_train_test(metadata, test_ratio=0.2)

# Further split train into train and validation
train_metadata, val_metadata = split_train_val(train_metadata, val_ratio=0.15)

'''
### TEMPORARY IMPLEMENTATION (COMMENT OUT WHEN DOING REAL TRAINING SET)
# TEMP: Reduce train set size for faster processing from 9295 samples to 1000
train_metadata = train_metadata.sample(n=1000, random_state=42)
'''

print(f" Train samples: {len(train_metadata)}")
print(f" Validation samples: {len(val_metadata)}")
print(f" Test samples: {len(test_metadata)}")

### Extract only relevant rows (protein coding genes)
from scripts.data_loader import get_protein_coding_gene_indices
protein_coding_indices = get_protein_coding_gene_indices("data/raw/human_gene_v2.5.h5")

'''
### TEMPORARY IMPLEMENTATION (COMMENT OUT WHEN DOING REAL TRAINING SET)
# Only for testing we will use ony 500 genes
protein_coding_indices = protein_coding_indices[:500]
'''

### Extracting an saving the data sets
'''

### Section where I create the Train, validation and test datasets

# Extract and save training set
print("About to extract train_data.npz...", flush=True)
extract_and_save_expression_matrix(
    filepath="data/raw/human_gene_v2.5.h5",
    gene_symbols=gene_symbols,
    sample_metadata=train_metadata,
    gene_row_indices=protein_coding_indices,
    target_gene="IRF2BP2",
    save_filename="train_data.npz"
)

# Extract and save validation set
print("About to extract val_data.npz...", flush=True)
extract_and_save_expression_matrix(
    filepath="data/raw/human_gene_v2.5.h5",
    gene_symbols=gene_symbols,
    sample_metadata=val_metadata,
    gene_row_indices=protein_coding_indices,
    target_gene="IRF2BP2",
    save_filename="val_data.npz"
)

# Extract and save test set
print("About to extract test_data.npz...", flush=True)
extract_and_save_expression_matrix(
    filepath="data/raw/human_gene_v2.5.h5",
    gene_symbols=gene_symbols,
    sample_metadata=test_metadata,
    gene_row_indices=protein_coding_indices,
    target_gene="IRF2BP2",
    save_filename="test_data.npz"
)

# STRUCTURE FROM X = (Rows = valid samples, Columns = protein coding genes)
# STRUCTURE FROM y = expression value of the IRF2BP2 gene for each of the valid samples

'''
