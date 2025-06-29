# scripts/interpretation.py

import torch
import numpy as np

def compute_gradient_x_input(model, data_loader, device, top_k_lowest=5):
    """
    Compute Gradient × Input scores for the samples with lowest predicted IRF2BP2.
    """
    model.eval()

    all_inputs = []
    all_preds = []

    # First collect all predictions
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).squeeze()
            all_inputs.append(X_batch.cpu())
            all_preds.append(preds.cpu())

    # Concatenate all batches
    all_inputs = torch.cat(all_inputs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    # Find indices of top_k_lowest predictions
    _, lowest_indices = torch.topk(all_preds, top_k_lowest, largest=False)

    # provides a 1D tensor of indices corresponding to the models with the lowest predictions of IRF2BP2

    selected_inputs = all_inputs[lowest_indices]

    # Now compute gradient x input for these selected inputs
    attributions = []

    for input_tensor in selected_inputs:
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        pred = model(input_tensor).squeeze()
        pred.backward()

        grad = input_tensor.grad.detach()  # gradient of output w.r.t input
        attribution = (grad * input_tensor).squeeze(0).cpu()  # gradient × input
        attributions.append(attribution)

    # Average across all selected samples
    avg_attribution = torch.stack(attributions).mean(dim=0)

    return avg_attribution.detach().numpy() # [num_genes]
