from sklearn.metrics import r2_score

def adjusted_r2_score(y_true, y_pred, p):
    """
    Compute adjusted R² score.
    
    Parameters:
        y_true (ndarray): True target values
        y_pred (ndarray): Predicted values
        p (int): Number of predictors (input features)

    Returns:
        float: Adjusted R² score
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def train_and_interpret_log_scaled_models(
    model_configs,
    train_loader,
    val_loader,
    test_loader,
    input_dim,
    gene_symbols,
    protein_coding_indices,
    device,
    save_dir="outputs"
):
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import pandas as pd
    from scripts.model import GeneExpressionRegressor, LinearWithActivation, LinearRegressor
    from scripts.model import ClosedFormLinearRegression
    from scripts.interpretation import compute_gradient_x_input
    from sklearn.metrics import r2_score

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for model_config in model_configs:
        hidden_str = "_".join([str(h) for h in model_config.get("hidden_dims", [])])
        dropout_str = f"d{model_config.get('dropout', 0.0)}" 
        extra_tags = []
        if model_config.get("batchnorm"):
            extra_tags.append("bn")
        if model_config.get("residual"):
            extra_tags.append("res")

        extra_str = "_" + "_".join(extra_tags) if extra_tags else ""
        output_tag = f"h{hidden_str}_{dropout_str}{extra_str}"
        print(f"\n==== Training model: {output_tag} ====")
        print(f"Architecture: {model_config}")

        # === Model instantiation logic ===
        model_type = model_config.get("type", "mlp")

        if model_type == "linear_closed":
            model = ClosedFormLinearRegression(
                input_dim=input_dim,
                add_bias=model_config.get("bias", True)
            )
        elif model_type == "linear_relu":
            model = LinearWithActivation(
                input_dim=input_dim,
                act=model_config.get("act", "relu")
            ).to(device)
        elif model_type == "linear":
            model = LinearRegressor(input_dim=input_dim).to(device)
        else:
            model = GeneExpressionRegressor(input_dim=input_dim, config=model_config).to(device)


        criterion = nn.MSELoss()
        model_type = model_config.get("type", "mlp")

        if model_type != "linear_closed":
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            num_epochs = 50
        best_val_loss = float("inf")
        train_losses, val_losses = [], []
        patience = 10
        no_improve_epochs = 0

        best_model_path = os.path.join("models", f"model_{output_tag}.pth")

        if model_type == "linear_closed":
            X_train, y_train = [], []
            for X, y in train_loader:
                X_train.append(X)
                y_train.append(y)
            X_train = torch.cat(X_train, dim=0)
            y_train = torch.cat(y_train, dim=0)
            model.fit(X_train, y_train)

            train_loss = torch.mean((model.predict(X_train) - y_train) ** 2).item()
            val_loss = 0.0
            for X, y in val_loader:
                preds = model.predict(X)
                val_loss += torch.mean((preds - y) ** 2).item()
            val_loss /= len(val_loader)

            print(f"[{output_tag}] Closed-form fit - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Simulate early stopping save
            best_model = model
            best_model_path = os.path.join("models", f"model_{output_tag}.npy")
            np.save(best_model_path, best_model.w)
        else:
            # === Gradient-based training ===
            best_val_loss = float("inf")
            train_losses, val_losses = [], []
            patience = 10
            no_improve_epochs = 0

            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(X).squeeze(), y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(device), y.to(device)
                        val_loss += criterion(model(X).squeeze(), y).item()
                val_loss /= len(val_loader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"[{output_tag}] Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved at Epoch {epoch+1}")
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

        # === Loss Plot ===
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.title(f"Loss Curve - {output_tag}")
        plt.xlabel("Epochs")
        plt.ylabel("MSE (log1p scale)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"loss_plot_{output_tag}.png"))
        plt.close()

        # === Load best model ===
        model_type = model_config.get("type", "mlp")

        if model_type == "linear_closed":
            best_model = ClosedFormLinearRegression(
                input_dim=input_dim,
                add_bias=model_config.get("bias", True)
            )
            best_model.w = np.load(best_model_path)
        else:
            if model_type == "linear_relu":
                best_model = LinearWithActivation(
                    input_dim=input_dim,
                    act=model_config.get("act", "relu")
                ).to(device)
            elif model_type == "linear":
                best_model = LinearRegressor(input_dim=input_dim).to(device)
            else:
                best_model = GeneExpressionRegressor(input_dim=input_dim, config=model_config).to(device)

            best_model.load_state_dict(torch.load(best_model_path))
            best_model.eval()

        # === Test Evaluation ===
        all_preds, all_targets = [], []
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                if model_type == "linear_closed":
                    pred = best_model.predict(X)
                else:
                    X, y = X.to(device), y.to(device)
                    pred = best_model(X).squeeze()
                test_loss += criterion(pred, y).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        test_loss /= len(test_loader)
        print(f"[{output_tag}] Test Loss (log1p scale): {test_loss:.4f}")
        
        # === Compute R2 and adjusted R2 in log1p scale ===
        y_true_concat = np.concatenate(all_targets)
        y_pred_concat = np.concatenate(all_preds)
        r2 = r2_score(y_true_concat, y_pred_concat)
        adj_r2 = adjusted_r2_score(y_true_concat, y_pred_concat, p=input_dim)

        print(f"[{output_tag}] R² Score (log1p scale): {r2:.4f}")
        print(f"[{output_tag}] Adjusted R² Score (log1p scale): {adj_r2:.4f}")


        # === Convert predictions back to raw TPM ===
        all_preds = np.expm1(np.concatenate(all_preds))
        all_targets = np.expm1(np.concatenate(all_targets))

        # === Prediction Plot ===
        plt.figure(figsize=(7, 7))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel("True IRF2BP2 TPM")
        plt.ylabel("Predicted TPM")
        plt.title(f"Prediction vs Truth (TPM) - {output_tag}")
        plt.savefig(os.path.join(save_dir, f"pred_scatter_{output_tag}.png"))
        plt.close()
        # === Gradient × Input Attribution (only for neural nets) ===
        if model_type != "linear_closed":
            attributions = compute_gradient_x_input(best_model, test_loader, device, top_k_lowest=5)

            # === Save full attribution scores for all genes ===
            if len(protein_coding_indices) != len(attributions):
                raise ValueError(f"Mismatch: {len(protein_coding_indices)} indices vs {len(attributions)} attributions")
            all_gene_names = [gene_symbols[i] for i in protein_coding_indices]

            df_all = pd.DataFrame({
                "Gene": all_gene_names,
                "Attribution_Score": attributions
            })
            os.makedirs(os.path.join(save_dir, "attributions_top_genes"), exist_ok=True)
            df_all.to_csv(os.path.join(save_dir, "attributions_top_genes", f"all_gene_attributions_{output_tag}.csv"), index=False)


            top_neg_idx = np.argsort(attributions)[:30]
            top_pos_idx = np.argsort(attributions)[-30:]

            neg_genes = [gene_symbols[protein_coding_indices[i]] for i in top_neg_idx]
            pos_genes = [gene_symbols[protein_coding_indices[i]] for i in top_pos_idx]

            df_neg = pd.DataFrame({"Gene": neg_genes, "Attribution_Score": attributions[top_neg_idx]})
            df_pos = pd.DataFrame({"Gene": pos_genes, "Attribution_Score": attributions[top_pos_idx]})
            df_neg.to_csv(os.path.join(save_dir, f"top_negative_genes_{output_tag}.csv"), index=False)
            df_pos.to_csv(os.path.join(save_dir, f"top_positive_genes_{output_tag}.csv"), index=False)

            # === Attribution Bar Plot ===
            combined = list(zip(neg_genes + pos_genes, np.concatenate((attributions[top_neg_idx], attributions[top_pos_idx]))))
            combined_sorted = sorted(combined, key=lambda x: x[1])
            sorted_genes, sorted_scores = zip(*combined_sorted)

            plt.figure(figsize=(10, 10))
            plt.barh(range(len(sorted_genes)), sorted_scores,
                        color=["red" if s < 0 else "green" for s in sorted_scores])
            plt.yticks(range(len(sorted_genes)), sorted_genes)
            plt.axvline(0, color="black", linestyle="--")
            plt.title(f"Top Genes - {output_tag}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"attribution_bar_plot_{output_tag}.png"))
            plt.close()



