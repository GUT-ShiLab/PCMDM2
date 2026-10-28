import os
import sys
from math import sqrt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from models.regressor_MLP import SimpleConvMLPRegressor, MLPRegressor, CombinedModel
from models.combined_model import train_epoch, evaluate
from losses.losses import MSELossNoContrastive
from data.dataset import MultiOmicsDataset


# ========================== Dataset paths ==========================

metadata_path = "../data/IBD/CLR_unfiltered_data/metadata_IBD.csv"
phylogeny_microbiome_path = "../data/IBD/DL_data/microbiome_Train_CLR.csv"
clr_microbiome_path       = "../data/IBD/CLR_unfiltered_data/micro_train_clr.csv"
metabolome_path = "../data/IBD/CLR_unfiltered_data/metab_train_clr.csv"
external_phylogeny_microbiome_path = "../data/IBD/DL_data/microbiome_Test_CLR.csv"
external_clr_microbiome_path       = "../data/IBD/CLR_unfiltered_data/micro_test_clr.csv"
external_metabolome_path = "../data/IBD/CLR_unfiltered_data/metab_test_clr.csv"

# ========================== Check file existence ==========================
for path in [metadata_path, phylogeny_microbiome_path, clr_microbiome_path, metabolome_path]:
    assert os.path.exists(path), f"ERROR: File does not exist: {path}"

if external_phylogeny_microbiome_path:
    assert os.path.exists(external_phylogeny_microbiome_path), f"ERROR: External phylogeny microbiome file does not exist: {external_phylogeny_microbiome_path}"
if external_clr_microbiome_path:
    assert os.path.exists(external_clr_microbiome_path), f"ERROR: External clr microbiome file does not exist: {external_clr_microbiome_path}"
if external_metabolome_path:
    assert os.path.exists(external_metabolome_path), f"ERROR: External metabolome file does not exist: {external_metabolome_path}"


# ========================== Hyperparameters ==========================
batch_size = 128
num_epochs = 1000
early_stop_patience = 30  # Early stopping: stop if training loss does not improve for N epochs
learning_rate = 1e-3
weight_decay = 1e-3
random_seed = 123

conv_dims = [32, 64, 128]
mlp_dims = [512, 256]
parallel_mlp_dims = [512, 256]
final_mlp_dims = [512, 256]
kernel_size = 5
dropout_rate = 0.4

print("========================== Current Hyperparameters ==========================")
print(f"batch_size: {batch_size}")
print(f"num_epochs: {num_epochs}")
print(f"early_stop_patience: {early_stop_patience}")
print(f"learning_rate: {learning_rate}")
print(f"weight_decay: {weight_decay}")
print(f"random_seed: {random_seed}")
print(f"conv_dims (CNN channels): {conv_dims}")
print(f"mlp_dims (MLP after CNN): {mlp_dims}")
print(f"parallel_mlp_dims (parallel MLP): {parallel_mlp_dims}")
print(f"final_mlp_dims (final MLP): {final_mlp_dims}")
print(f"kernel_size: {kernel_size}")
print(f"dropout_rate: {dropout_rate}")
print("============================================================================")


# ========================== Device ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== Load dataset ==========================
dataset = MultiOmicsDataset(
    phylogeny_microbiome_path,
    clr_microbiome_path,
    metabolome_path,
    metadata_path=None,
    external_phylogeny_microbiome_path=external_phylogeny_microbiome_path,
    external_clr_microbiome_path=external_clr_microbiome_path,
    external_metabolome_path=external_metabolome_path
)

# Get feature dimensions
phylogeny_microbiome_data = pd.read_csv(phylogeny_microbiome_path, index_col=0)
clr_microbiome_data = pd.read_csv(clr_microbiome_path, index_col=0)
metabolome_data = pd.read_csv(metabolome_path, index_col=0)

conv_input_dim = phylogeny_microbiome_data.shape[0]      # number of phylogeny-informed microbiome features
mlp_input_dim = clr_microbiome_data.shape[0]             # number of clr microbiome features
metabolome_output_dim = metabolome_data.shape[0]         # number of metabolite targets

print("Number of phylogeny-informed microbiome features:", conv_input_dim)
print("Number of clr microbiome features:", mlp_input_dim)
print("Number of metabolite targets:", metabolome_output_dim)


# ========================== Cross-validation ==========================
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

fold_results = {f"Fold_{i}_Scc": [] for i in range(5)}
for i in range(5):
    fold_results[f"Fold_{i}_RMSE"] = []
    fold_results[f"Fold_{i}_MAAPE"] = []
    fold_results[f"Fold_{i}_NMSE"] = []

metabolite_names = None

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Training fold {fold + 1}...")

    # IMPORTANT: sampler must be a Sampler object, not raw indices
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    model = CombinedModel(
        conv_input_dim=conv_input_dim,
        mlp_input_dim=mlp_input_dim,
        conv_dims=conv_dims,
        mlp_dims=mlp_dims,
        parallel_mlp_dims=parallel_mlp_dims,
        final_mlp_dims=final_mlp_dims,
        output_dim=metabolome_output_dim,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = MSELossNoContrastive().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Fold {fold}, Epoch {epoch}, Loss: {train_loss}")

        # Note: this is train-loss-based early stopping (not validation-based)
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at fold {fold}, epoch {epoch}")
                break

    metabolite_names, fold_scc, _, _, fold_rmse, fold_maape, fold_nmse = evaluate(model, test_loader, device)
    fold_results[f"Fold_{fold}_Scc"] = fold_scc
    fold_results[f"Fold_{fold}_RMSE"] = fold_rmse
    fold_results[f"Fold_{fold}_MAAPE"] = fold_maape
    fold_results[f"Fold_{fold}_NMSE"] = fold_nmse


result_df = pd.DataFrame({"Metabolites": metabolite_names})

# Add average metric columns across folds
for metric in ["Scc", "RMSE", "MAAPE", "NMSE"]:
    result_df[f"Average_{metric}"] = np.mean([fold_results[f"Fold_{i}_{metric}"] for i in range(5)], axis=0)

# Sort by average Spearman correlation
result_df = result_df.sort_values(by="Average_Scc", ascending=False)

# Summary statistics
avg_spearman_corr_all = np.mean(result_df["Average_Scc"])
avg_rmse_all = np.mean(result_df["Average_RMSE"])
avg_maape_all = np.mean(result_df["Average_MAAPE"])
avg_nmse_all = np.mean(result_df["Average_NMSE"])

top_50_spearman = result_df["Average_Scc"].head(50)
avg_spearman_corr_top50 = np.mean(top_50_spearman)

num_metabolites_ge_0_3 = (result_df["Average_Scc"] >= 0.3).sum()
proportion_ge_0_3 = num_metabolites_ge_0_3 / metabolome_output_dim

print("========================================= Five-Fold CV =========================================")
print(f"Five-Fold Average Spearman Correlation (All Metabolites): {avg_spearman_corr_all:.4f}")
print(f"Five-Fold Average Spearman Correlation (Top 50 Metabolites): {avg_spearman_corr_top50:.4f}")
print(f"Number of metabolites with Spearman correlation >= 0.3: {num_metabolites_ge_0_3}")
print(f"Proportion of metabolites with Spearman correlation >= 0.3: {proportion_ge_0_3:.2%}")
print(f"Five-Fold Average MAAPE (All Metabolites): {avg_maape_all:.4f}")
# print(f"Five-Fold Average RMSE (All Metabolites): {avg_rmse_all:.4f}")
# print(f"Five-Fold Average NMSE (All Metabolites): {avg_nmse_all:.4f}")


# ========================== External validation ==========================
external_phylogeny_microbiome, external_clr_microbiome, external_metabolome = dataset.get_external_validation_data()

external_phylogeny_microbiome = torch.tensor(external_phylogeny_microbiome.values, dtype=torch.float32).to(device)
external_clr_microbiome = torch.tensor(external_clr_microbiome.values, dtype=torch.float32).to(device)
external_metabolome = torch.tensor(external_metabolome.values, dtype=torch.float32).to(device)

print(f"External phylogeny microbiome shape: {external_phylogeny_microbiome.shape}")
print(f"External clr microbiome shape: {external_clr_microbiome.shape}")
print(f"External metabolome shape: {external_metabolome.shape}")


# Retrain final model on the full training set
final_model = CombinedModel(
    conv_input_dim=conv_input_dim,
    mlp_input_dim=mlp_input_dim,
    conv_dims=conv_dims,
    mlp_dims=mlp_dims,
    parallel_mlp_dims=parallel_mlp_dims,
    final_mlp_dims=final_mlp_dims,
    output_dim=metabolome_output_dim,
    kernel_size=kernel_size,
    dropout_rate=dropout_rate
).to(device)

final_criterion = MSELossNoContrastive().to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
final_train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

best_loss = float("inf")
patience_counter = 0

for epoch in range(num_epochs):
    final_train_loss = train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, device)
    if final_train_loss < best_loss:
        best_loss = final_train_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping for final model at epoch {epoch}")
            break


# Predict on external validation set
with torch.no_grad():
    final_model.eval()
    external_predictions = final_model(external_phylogeny_microbiome, external_clr_microbiome).cpu().numpy()
    external_targets = external_metabolome.cpu().numpy()

    spearman_corr_external = [
        spearmanr(external_predictions[:, i], external_targets[:, i])[0]
        for i in range(external_targets.shape[1])
    ]
    external_rmse = [
        sqrt(np.mean((external_predictions[:, i] - external_targets[:, i]) ** 2))
        for i in range(external_targets.shape[1])
    ]
    external_maape = [
        np.mean(np.arctan(np.abs((external_targets[:, i] - external_predictions[:, i]) / external_targets[:, i]))) * 2 / np.pi
        for i in range(external_targets.shape[1])
    ]
    external_nmse = [
        (np.mean((external_targets[:, i] - external_predictions[:, i]) ** 2) / np.var(external_targets[:, i]))
        if np.var(external_targets[:, i]) > 0 else np.nan
        for i in range(external_targets.shape[1])
    ]

    external_results_df = pd.DataFrame({
        "Metabolites": metabolite_names,
        "Spearman_Correlation": spearman_corr_external,
        "RMSE": external_rmse,
        "MAAPE": external_maape,
        "NMSE": external_nmse
    })

    external_metabolome_predictions = "../results/external_metabolome_predictions_pcmdm2.csv"
    pd.DataFrame(external_predictions, columns=metabolite_names).to_csv(external_metabolome_predictions, index=False)

    num_metabolites_ge_0_3 = sum(np.array(spearman_corr_external) >= 0.3)
    proportion_ge_0_3 = num_metabolites_ge_0_3 / len(spearman_corr_external)
    top_50_avg_spearman = external_results_df.nlargest(50, "Spearman_Correlation")["Spearman_Correlation"].mean()

    print("========================================= External Test Prediction =========================================")
    avg_spearman_corr_external_all = np.mean(spearman_corr_external)
    avg_maape_external_all = np.mean(external_maape)

    print(f"External Validation Average Spearman Correlation (All Metabolites): {avg_spearman_corr_external_all}")
    print(f"External Validation Average Spearman Correlation (Top 50 Metabolites): {top_50_avg_spearman}")
    print(f"Number of metabolites with Spearman correlation >= 0.3 (External): {num_metabolites_ge_0_3}")
    print(f"Proportion of metabolites with Spearman correlation >= 0.3 (External): {proportion_ge_0_3:.2%}")
    print(f"External Validation Average MAAPE (All Metabolites): {avg_maape_external_all}")
