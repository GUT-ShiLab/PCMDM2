import torch
import numpy as np
from scipy.stats import spearmanr

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for phylogeny_microbiome, clr_microbiome, metabolome in train_loader:
        phylogeny_microbiome = phylogeny_microbiome.to(device)
        clr_microbiome = clr_microbiome.to(device)
        metabolome = metabolome.to(device)

        # Forward pass
        predictions = model(phylogeny_microbiome, clr_microbiome)

        # Compute loss (MSE)
        loss = criterion(predictions, metabolome)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def compute_metrics(y_pred, y_true):
    """
    Compute RMSE, MAAPE, and NMSE.
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))


    abs_percentage_error = np.abs((y_true - y_pred) / y_true)
    maape = np.mean(np.arctan(abs_percentage_error)) * 2 / np.pi

    mse_val = np.mean((y_true - y_pred) ** 2)
    var_y = np.var(y_true)
    nmse = mse_val / var_y if var_y > 0 else np.nan  # Avoid division by zero

    return rmse, maape, nmse


def evaluate(model, dataloader, device):
    """
    Evaluate the model on the test set for a single fold.
    Returns:
        metabolite_names, spearman_corrs, predictions, targets, rmse_vals, maape_vals, nmse_vals
    """
    model.eval()
    predictions = []
    targets = []

    # Get metabolite names
    metabolite_names = dataloader.dataset.metabolome_data.columns

    with torch.no_grad():
        for phylogeny_microbiome, clr_microbiome, metabolome in dataloader:
            phylogeny_microbiome = phylogeny_microbiome.to(device)
            clr_microbiome = clr_microbiome.to(device)
            metabolome = metabolome.to(device)

            preds = model(phylogeny_microbiome, clr_microbiome)
            predictions.append(preds.cpu().numpy())
            targets.append(metabolome.cpu().numpy())

    # Convert lists to numpy arrays
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Compute Spearman correlation and other metrics for each metabolite
    spearman_corrs = []
    rmse_vals = []
    maape_vals = []
    nmse_vals = []

    for i in range(targets.shape[1]):  # Iterate over metabolites
        corr, _ = spearmanr(predictions[:, i], targets[:, i])
        spearman_corrs.append(corr)

        rmse, maape, nmse = compute_metrics(predictions[:, i], targets[:, i])
        rmse_vals.append(rmse)
        maape_vals.append(maape)
        nmse_vals.append(nmse)

    return metabolite_names, spearman_corrs, predictions, targets, rmse_vals, maape_vals, nmse_vals
