def train_and_interpret_all_models(
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
    from scripts.model import GeneExpressionRegressor
    from scripts.interpretation import compute_gradient_x_input
    from scripts.summarise_top_genes import summarise_top_gene_matrix

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for model_config in model_configs:
        hidden_str = "_".join([str(h) for h in model_config["hidden_dims"]])
        dropout_str = f"d{model_config['dropout']}" 
        extra_tags = []
        if model_config.get("batchnorm"):
            extra_tags.append("bn")
        if model_config.get("residual"):
           extra_tags.append("res")

        extra_str = "_" + "_".join(extra_tags) if extra_tags else ""
        output_tag = f"h{hidden_str}_{dropout_str}{extra_str}"
        print(f"\n==== Training model: {output_tag} ====")
        print(f"Architecture: {model_config}")

        model = GeneExpressionRegressor(input_dim=input_dim, config=model_config).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 50
        best_val_loss = float("inf")
        train_losses, val_losses = [], []
        patience = 10 # sets the amount of epochs after that when no improvement happens the training is cancelled
        no_improve_epochs = 0


        best_model_path = os.path.join("models", f"model_{output_tag}.pth")

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
        plt.ylabel("MSE")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"loss_plot_{output_tag}.png"))
        plt.close()

        # === Explicit re-instantiation for test ===
        best_model = GeneExpressionRegressor(input_dim=input_dim, config=model_config).to(device)
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.eval()

        # === Test Evaluation ===
        all_preds, all_targets = [], []
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = best_model(X).squeeze()
                test_loss += criterion(pred, y).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        test_loss /= len(test_loader)
        print(f"[{output_tag}] Test Loss: {test_loss:.4f}")

        # === Prediction Plot ===
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        plt.figure(figsize=(7, 7))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel("True IRF2BP2")
        plt.ylabel("Predicted")
        plt.title(f"Prediction vs Truth - {output_tag}")
        plt.savefig(os.path.join(save_dir, f"pred_scatter_{output_tag}.png"))
        plt.close()

        # === Gradient × Input Attribution ===
        attributions = compute_gradient_x_input(best_model, test_loader, device, top_k_lowest=5)

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

    