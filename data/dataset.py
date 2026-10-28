
import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiOmicsDataset(Dataset):
    """Custom dataset for microbiome and metabolome data without labels."""
    def __init__(self, phylogeny_microbiome_path, clr_microbiome_path, metabolome_path, metadata_path=None,
                 external_phylogeny_microbiome_path=None, external_clr_microbiome_path=None, external_metabolome_path=None):
        """
        Args:
            phylogeny_microbiome_path (str): Path to phylogeny-informed microbiome data (CSV).
            clr_microbiome_path (str): Path to clr microbiome data (CSV).
            metabolome_path (str): Path to metabolome data (CSV).
            metadata_path (str, optional): Path to metadata (CSV), not used in this version.
            external_phylogeny_microbiome_path (str, optional): Path to external phylogeny-informed microbiome test set (CSV).
            external_clr_microbiome_path (str, optional): Path to external clr microbiome test set (CSV).
            external_metabolome_path (str, optional): Path to external metabolome test set (CSV).
        """
        # Load training data
        self.phylogeny_microbiome_data = pd.read_csv(phylogeny_microbiome_path, index_col=0)
        self.clr_microbiome_data = pd.read_csv(clr_microbiome_path, index_col=0)
        self.metabolome_data = pd.read_csv(metabolome_path, index_col=0)

        # Align samples in the training set
        self.samples = self.phylogeny_microbiome_data.columns
        self.phylogeny_microbiome_data = self.phylogeny_microbiome_data.T
        self.clr_microbiome_data = self.clr_microbiome_data.T
        self.metabolome_data = self.metabolome_data.T

        #Load external validation data (if provided)
        self.external_phylogeny_microbiome_data = None
        self.external_clr_microbiome_data = None
        self.external_metabolome_data = None

        if external_phylogeny_microbiome_path and external_clr_microbiome_path and external_metabolome_path:
            external_phylogeny_microbiome_data = pd.read_csv(external_phylogeny_microbiome_path, index_col=0)
            external_clr_microbiome_data = pd.read_csv(external_clr_microbiome_path, index_col=0)
            external_metabolome_data = pd.read_csv(external_metabolome_path, index_col=0)

            print(f"External phylogeny microbiome shape before: {external_phylogeny_microbiome_data.shape}")
            print(f"External clr microbiome shape before: {external_clr_microbiome_data.shape}")

            # Filter external features to keep only those present in the training set
            self.external_phylogeny_microbiome_data = external_phylogeny_microbiome_data.loc[self.phylogeny_microbiome_data.columns].T
            self.external_clr_microbiome_data = external_clr_microbiome_data.loc[self.clr_microbiome_data.columns].T
            self.external_metabolome_data = external_metabolome_data.loc[self.metabolome_data.columns].T

            print(f"External phylogeny microbiome shape after: {self.external_phylogeny_microbiome_data.shape}")
            print(f"External clr microbiome shape after: {self.external_clr_microbiome_data.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        phylogeny_microbiome = torch.tensor(self.phylogeny_microbiome_data.loc[sample].values, dtype=torch.float32)
        clr_microbiome = torch.tensor(self.clr_microbiome_data.loc[sample].values, dtype=torch.float32)
        metabolome = torch.tensor(self.metabolome_data.loc[sample].values, dtype=torch.float32)
        return phylogeny_microbiome, clr_microbiome, metabolome

    def get_external_validation_data(self):
        """Return external validation set."""
        if (self.external_phylogeny_microbiome_data is not None
            and self.external_clr_microbiome_data is not None
            and self.external_metabolome_data is not None):
            return (self.external_phylogeny_microbiome_data,
                    self.external_clr_microbiome_data,
                    self.external_metabolome_data)
        else:
            raise ValueError("External validation data is not provided.")