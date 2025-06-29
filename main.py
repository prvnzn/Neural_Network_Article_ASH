from scripts.train_and_interpret import train_and_interpret_all_models
from scripts.data_loader import extract_metadata, get_protein_coding_gene_indices
from scripts.gene_expression_dataset import GeneExpressionDataset
from torch.utils.data import DataLoader
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load metadata
gene_symbols, source_names, singlecell_prob = extract_metadata("data/raw/human_gene_v2.5.h5")
protein_coding_indices = get_protein_coding_gene_indices("data/raw/human_gene_v2.5.h5")
protein_coding_indices = [i for i in protein_coding_indices if gene_symbols[i] != "IRF2BP2"]


# Load log1p-transformed data
train_loader = DataLoader(GeneExpressionDataset("data/processed/train_data_log1p.npz"), batch_size=32, shuffle=True)
val_loader = DataLoader(GeneExpressionDataset("data/processed/val_data_log1p.npz"), batch_size=32, shuffle=False)
test_loader = DataLoader(GeneExpressionDataset("data/processed/test_data_log1p.npz"), batch_size=32, shuffle=False)
input_dim = train_loader.dataset.X.shape[1]


# initial model configurations
'''

### NON-LOG SCALED
# 1.0
model_configs = [
    {"hidden_dims": [128, 64], "dropout": 0.1},  # Balanced architecture, likely optimal
    {"hidden_dims": [256, 128], "dropout": 0.3},  # Mid-depth model with stronger dropout
    {"hidden_dims": [512, 256, 64], "dropout": 0.3},  # High-capacity model, risk of overfitting
    {"hidden_dims": [64, 32], "dropout": 0.1}, # New architecture based on loss function analysis
    {"hidden_dims": [32], "dropout": 0.1}  # Very shallow model, tests underfitting
]


# 2.0
model_configs = [
    {"hidden_dims": [32], "dropout": 0.1},              # Best-performing model so far
    {"hidden_dims": [64, 32], "dropout": 0.1},          # Slightly deeper, still compact
    {"hidden_dims": [64, 64], "dropout": 0.1},          # Symmetric moderate-depth model
    {"hidden_dims": [32, 32, 16], "dropout": 0.05}      # Deeper, small units with minimal dropout
]

# 3.0
model_configs = [
    {"hidden_dims": [32], "dropout": 0.1},                        # Original best
    {"hidden_dims": [64, 32], "dropout": 0.1},                    # Slightly deeper
    {"hidden_dims": [32], "dropout": 0.1, "residual": True},      # Residual skip connection
    {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True}, # BatchNorm version
    {"hidden_dims": [32, 32, 16], "dropout": 0.05, "batchnorm": True}  # Deeper with BatchNorm
]


# Promosing configuration: {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True}
# 4.0
model_configs = [
    {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True},       # best so far
    {"hidden_dims": [128, 64], "dropout": 0.1, "batchnorm": True},      # slightly wider
    {"hidden_dims": [64, 64], "dropout": 0.1, "batchnorm": True},       # symmetric, slightly deeper
    {"hidden_dims": [64, 32], "dropout": 0.05, "batchnorm": True},      # same structure, lower dropout
    {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True, "residual": True}  # optional: add skip connection
]
    
# 5.0
model_configs = [
    {"hidden_dims": [128, 64, 32], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [64, 64, 32], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [64, 64, 32], "dropout": 0.05, "residual": True}, 
]


### LOG SCALED CONFIGURATIONS(BETTER)
# 1.0
model_configs = [
    {"hidden_dims": [128, 64, 32], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [64, 32], "dropout": 0.1},
    {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True}, 
    {"hidden_dims": [32, 32, 16], "dropout": 0.05, "batchnorm": True}, 
]


# 2.0 
model_configs = [
    {"hidden_dims": [128, 64], "dropout": 0.1, "batchnorm": True}, 
    {"hidden_dims": [32], "dropout": 0.1}, 
    {"hidden_dims": [128, 64, 32], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [64, 32, 16], "dropout": 0.05, "batchnorm": True}, 

]



# 3.0 
model_configs = [
    {"hidden_dims": [512, 256, 64], "dropout": 0.3, "batchnorm": True},
    {"hidden_dims": [512, 256, 64], "dropout": 0.1, "batchnorm": True}, # promising data
    {"hidden_dims": [256, 64], "dropout": 0.3, "batchnorm": True},
    {"hidden_dims": [256, 64], "dropout": 0.1, "batchnorm": True}
]


# 4.0
model_configs = [
    {"hidden_dims": [512, 256, 64], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [1024, 512, 256, 64], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [1024, 512, 256], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [512, 256, 64, 32], "dropout": 0.1, "batchnorm": True}
]

# 5 .0
model_configs = [
    {"hidden_dims": [512, 256, 64, 32], "dropout": 0.05, "batchnorm": True}, 
    {"hidden_dims": [512, 256, 64, 32, 16], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [512, 256, 64, 32, 16], "dropout": 0.05, "batchnorm": True} # really good model 
]

# 6.0
model_configs = [
    {"hidden_dims": [512, 256, 64, 32, 16, 8], "dropout": 0.05, "batchnorm": True}, 
    {"hidden_dims": [256, 64, 32, 16], "dropout": 0.05, "batchnorm": True}, 
    {"hidden_dims": [1024, 512, 256, 64, 32], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [1024, 512, 256, 64, 32, 16], "dropout": 0.05, "batchnorm": True}
]

# 7.0
model_configs = [
    {"hidden_dims": [512, 256, 64], "dropout": 0.1, "batchnorm": True}, 
    {"hidden_dims": [512, 256, 64], "dropout": 0.15, "batchnorm": True},
    {"hidden_dims": [1028, 512, 256, 64], "dropout": 0.1, "batchnorm": True},
]

# 8.0
model_configs = [
    {"hidden_dims": [512, 256, 64], "dropout": 0.2, "batchnorm": True},
    {"hidden_dims": [1028, 512, 256, 64], "dropout": 0.2, "batchnorm": True},
]

# 9.0
model_configs = [
    {"hidden_dims": [1024, 512, 256, 64, 32], "dropout": 0.05, "batchnorm": True}, 
    {"hidden_dims": [256, 64, 32, 16], "dropout": 0.2, "batchnorm": True},
    {"hidden_dims": [512, 256, 64, 32, 16], "dropout": 0.15, "batchnorm": True}
]

# 10.0
model_configs = [
    {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True}, # good model
    {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True, "residual": True}, 
    {"hidden_dims": [64, 32, 16], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [64, 32, 16], "dropout": 0.1, "batchnorm": True, "residual": True},
    {"hidden_dims": [64, 32, 16], "dropout": 0.05, "batchnorm": True},
]

# 11.0
model_configs = [
    {"hidden_dims": [64, 32], "dropout": 0.05, "batchnorm": True}, 
    {"hidden_dims": [64, 32], "dropout": 0.01, "batchnorm": True}, 
    {"hidden_dims": [32, 16], "dropout": 0.1, "batchnorm": True}, # good model
    {"hidden_dims": [32, 16], "dropout": 0.05, "batchnorm": True}
]

# 12.0
model_configs = [
    {"hidden_dims": [32], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [32, 16], "dropout": 0.2, "batchnorm": True},
    {"hidden_dims": [32, 16, 8], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [32, 16, 8], "dropout": 0.05, "batchnorm": True},
]

# 13.0
model_configs = [
    {"hidden_dims": [64], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [64], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [64], "dropout": 0.2, "batchnorm": True},
    {"hidden_dims": [64], "dropout": 0.1},
]

'''

'''
# Best candidates for summary matrix: 
model_configs = [
    {"hidden_dims": [512, 256, 64], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [512, 256, 64, 32], "dropout": 0.05, "batchnorm": True}, 
    {"hidden_dims": [64, 32], "dropout": 0.1, "batchnorm": True}, 
    {"hidden_dims": [32, 16], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [32], "dropout": 0.1, "batchnorm": True},
]

'''

# Round 2 model configurations
'''
# 1.0
model_configs = [
    {"hidden_dims": [64], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [32], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [32, 16], "dropout": 0.1, "batchnorm": True}
]


# 2.0 (need to adjust the parameters)
model_configs = [
    {"hidden_dims": [256], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [256], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [256], "dropout": 0.15, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.1, "batchnorm": True},
]


# 3.0
model_configs = [
    {"hidden_dims": [32], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [64], "dropout": 0.2, "batchnorm": True},
    {"hidden_dims": [16], "dropout": 0.3, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.3, "batchnorm": True},
]



# 4.0
model_configs = [
    {"hidden_dims": [128], "dropout": 0.35, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.2, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.01, "batchnorm": True},
    {"hidden_dims": [128], "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.5, "batchnorm": True},
]


# 5.0
model_configs = [
    {"hidden_dims": [128], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.09, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.08, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.07, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.06, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.04, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.03, "batchnorm": True},
]

# 6.0
model_configs = [
    {"hidden_dims": [512], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [512], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [512], "dropout": 0.15, "batchnorm": True},
    {"hidden_dims": [1024], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [1024], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [1024], "dropout": 0.15, "batchnorm": True},
]


# Closed Linear model configuration for comparison
model_configs = [
    {
        "type": "linear_closed",
        "bias": True
    }
]
'''

# Model configurations selected for manuscript submission
model_configs = [
    {"hidden_dims": [64], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.03, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.5, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.05, "batchnorm": True},
    {"hidden_dims": [128], "dropout": 0.35, "batchnorm": True},
    {"hidden_dims": [256], "dropout": 0.1, "batchnorm": True},
    {"hidden_dims": [512], "dropout": 0.1, "batchnorm": True},
]

'''
# Worst performing model setup
# Train,, evaluate and create outputs using non-log scaled TPM
train_and_interpret_all_models(
    model_configs=model_configs,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    input_dim=input_dim,
    gene_symbols=gene_symbols,
    protein_coding_indices=protein_coding_indices,
    device=device
)

'''

# Better performing model setup
# log scaled TPM for better handling of high variance
from scripts.train_and_interpret_log import train_and_interpret_log_scaled_models


train_and_interpret_log_scaled_models(
    model_configs,
    train_loader,
    val_loader,
    test_loader,
    input_dim,
    gene_symbols,
    protein_coding_indices,
    device,
    save_dir="outputs"
)
