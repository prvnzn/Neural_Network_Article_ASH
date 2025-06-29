# scripts/gene_expression_dataset.py

import numpy as np
import torch

class GeneExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file_path):
        """
        Loads gene expression data and labels (IRF2BP2 expression) from a .npz file.
        """
        # Load the .npz file
        data = np.load(npz_file_path)

        # Store the expression matrix (X) and labels (y) as numpy arrays
        self.X = data["X"]
        self.y = data["y"]

        # Convert them to PyTorch tensors immediately for faster training
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a single (X[idx], y[idx]) pair for a given index.
        """
        return self.X[idx], self.y[idx]
