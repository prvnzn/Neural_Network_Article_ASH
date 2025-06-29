import torch
import torch.nn as nn
import numpy as np

class GeneExpressionRegressor(nn.Module):
    def __init__(self, input_dim, config):
        """
        A flexible feedforward neural network for gene expression regression.

        input_dim: Number of input genes (features)
        config: Dictionary containing:
            - "hidden_dims": list of integers for hidden layer sizes (amount of neurons in each layer)
            - "dropout": float, dropout probability (optional) (randomly zeroes some of the activations)
            - "batchnorm": bool, whether to use batch normalization (optional) (normalizes activations)
            - "residual": bool, whether to use residual connections (optional) (skips layers)
        """
        super(GeneExpressionRegressor, self).__init__()

        hidden_dims = config["hidden_dims"]
        dropout_rate = config.get("dropout", 0.0)
        use_batchnorm = config.get("batchnorm", False)
        use_residual = config.get("residual", False)

        self.use_residual = use_residual and len(hidden_dims) == 1  # only allow residuals on one-layer models

        layers = []
        prev_dim = input_dim

        self.residual_layer = None
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            layers.append(linear)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            if self.use_residual:
                self.residual_layer = linear
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model[:-1](x)  # all layers except the final
        if self.use_residual:
            res = self.residual_layer(x)
            out = out + res
        return self.model[-1](out)
    

# Model for no hidden layer but with ReLU activation 
class LinearWithActivation(nn.Module):
    def __init__(self, input_dim, act="relu"):
        super(LinearWithActivation, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.activation = nn.ReLU() if act == "relu" else nn.Identity()

    def forward(self, x):
        return self.activation(self.fc(x)).squeeze()


import torch.nn.functional as F
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(ConvNet, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(dropout)

        # Placeholder for fc — actual dimensions will be inferred later
        self.fc = None

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, input_dim]
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten

        # Dynamically initialise fc on first pass
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], 1).to(x.device)

        x = self.dropout(x)
        return self.fc(x).squeeze()

    
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1, batchnorm=False, residual=False, act="relu"):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        self.residual = residual
        self.activation = nn.LeakyReLU() if act == "leaky" else nn.ReLU()

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x):
        out = self.hidden(x)
        if self.residual:
            out += x  # Assumes dimensions match (e.g. use only with 1 hidden layer where input_dim == h_dim)
        return self.output(out).squeeze()


class ClosedFormLinearRegression:
    def __init__(self, input_dim, add_bias=True):
        self.add_bias = add_bias
        self.w = None

    def fit(self, X, y):
        """
        X: torch.Tensor (n_samples, n_features)
        y: torch.Tensor (n_samples,)
        """
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy().reshape(-1, 1)

        if self.add_bias:
            X_np = np.hstack([np.ones((X_np.shape[0], 1)), X_np])

        self.w = np.linalg.pinv(X_np) @ y_np

    def predict(self, X):
        """
        X: torch.Tensor (n_samples, n_features)
        Returns: torch.Tensor (n_samples,)
        """
        X_np = X.cpu().numpy()
        if self.add_bias:
            X_np = np.hstack([np.ones((X_np.shape[0], 1)), X_np])
        y_pred = X_np @ self.w  # (n_samples, 1)
        return torch.tensor(y_pred.flatten(), dtype=torch.float32)


class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressor, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze()