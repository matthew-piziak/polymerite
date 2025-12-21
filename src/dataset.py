"""
PyTorch Dataset for polymer property prediction.

Loads SMILES and property values from CSV files and applies featurization.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from .featurization import PolymerFeaturizer, MorganFingerprintFeaturizer


class PolymerDataset(Dataset):
    """
    Dataset for polymer property prediction using molecular graphs.

    Loads SMILES and property values from CSV files and converts to PyTorch
    Geometric graphs using PolymerFeaturizer.
    """

    def __init__(
        self,
        csv_path: str,
        property_name: str,
        featurizer: Optional[PolymerFeaturizer] = None,
        smiles_column: str = "SMILES",
        property_column: str = "property_value",
        normalize: bool = True,
        transform=None,
    ):
        """
        Initialize polymer dataset.

        Args:
            csv_path: Path to CSV file with SMILES and property values
            property_name: Name of the property (for metadata)
            featurizer: PolymerFeaturizer instance (default: create new one)
            smiles_column: Name of column containing SMILES strings
            property_column: Name of column containing property values
            normalize: Whether to normalize property values (z-score)
            transform: Optional transform to apply to graphs
        """
        self.csv_path = Path(csv_path)
        self.property_name = property_name
        self.featurizer = featurizer or PolymerFeaturizer()
        self.smiles_column = smiles_column
        self.property_column = property_column
        self.transform = transform

        # Load data
        self.df = pd.read_csv(csv_path)

        # Check columns exist
        if smiles_column not in self.df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in {csv_path}")
        if property_column not in self.df.columns:
            raise ValueError(f"Column '{property_column}' not found in {csv_path}")

        # Remove rows with missing values
        self.df = self.df.dropna(subset=[smiles_column, property_column])

        # Normalization statistics
        self.mean = self.df[property_column].mean()
        self.std = self.df[property_column].std()
        self.normalize = normalize

        # Convert to graphs (cache for efficiency)
        self._prepare_graphs()

        print(f"Loaded {len(self)} samples from {self.csv_path.name}")
        if normalize:
            print(f"Property stats: mean={self.mean:.4f}, std={self.std:.4f}")

    def _prepare_graphs(self):
        """Pre-convert all SMILES to graphs for faster training."""
        self.graphs = []
        self.properties = []
        failed = 0

        for idx, row in self.df.iterrows():
            smiles = row[self.smiles_column]
            prop_value = row[self.property_column]

            # Convert SMILES to graph
            graph = self.featurizer.smiles_to_graph(smiles)

            if graph is not None:
                # Normalize property value if requested
                if self.normalize:
                    y = (prop_value - self.mean) / self.std
                else:
                    y = prop_value

                graph.y = torch.tensor([y], dtype=torch.float)
                graph.smiles = smiles

                self.graphs.append(graph)
                self.properties.append(prop_value)
            else:
                failed += 1

        if failed > 0:
            print(f"Warning: Failed to parse {failed} SMILES strings")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """Get a single graph."""
        graph = self.graphs[idx]

        if self.transform is not None:
            graph = self.transform(graph)

        return graph

    def denormalize(self, normalized_value: float) -> float:
        """Convert normalized property value back to original scale."""
        if self.normalize:
            return normalized_value * self.std + self.mean
        return normalized_value

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            'property': self.property_name,
            'n_samples': len(self),
            'mean': self.mean,
            'std': self.std,
            'min': min(self.properties),
            'max': max(self.properties),
        }


class PolymerFingerprintDataset(Dataset):
    """
    Dataset for polymer property prediction using Morgan fingerprints.

    This is useful for training traditional ML models (XGBoost, Random Forest)
    or simple MLPs.
    """

    def __init__(
        self,
        csv_path: str,
        property_name: str,
        featurizer: Optional[MorganFingerprintFeaturizer] = None,
        smiles_column: str = "SMILES",
        property_column: str = "property_value",
        normalize: bool = True,
    ):
        """
        Initialize fingerprint dataset.

        Args:
            csv_path: Path to CSV file with SMILES and property values
            property_name: Name of the property (for metadata)
            featurizer: MorganFingerprintFeaturizer instance
            smiles_column: Name of column containing SMILES strings
            property_column: Name of column containing property values
            normalize: Whether to normalize property values
        """
        self.csv_path = Path(csv_path)
        self.property_name = property_name
        self.featurizer = featurizer or MorganFingerprintFeaturizer()
        self.smiles_column = smiles_column
        self.property_column = property_column

        # Load data
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=[smiles_column, property_column])

        # Normalization
        self.mean = self.df[property_column].mean()
        self.std = self.df[property_column].std()
        self.normalize = normalize

        # Prepare fingerprints
        self._prepare_fingerprints()

        print(f"Loaded {len(self)} samples from {self.csv_path.name}")

    def _prepare_fingerprints(self):
        """Pre-compute all fingerprints."""
        self.fingerprints = []
        self.properties = []
        failed = 0

        for idx, row in self.df.iterrows():
            smiles = row[self.smiles_column]
            prop_value = row[self.property_column]

            fp = self.featurizer.smiles_to_fingerprint(smiles)

            if fp is not None:
                if self.normalize:
                    y = (prop_value - self.mean) / self.std
                else:
                    y = prop_value

                self.fingerprints.append(fp)
                self.properties.append(y)
            else:
                failed += 1

        if failed > 0:
            print(f"Warning: Failed to parse {failed} SMILES strings")

        # Convert to arrays
        self.fingerprints = np.array(self.fingerprints, dtype=np.float32)
        self.properties = np.array(self.properties, dtype=np.float32)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.fingerprints)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single fingerprint and property value."""
        x = torch.from_numpy(self.fingerprints[idx])
        y = torch.tensor([self.properties[idx]], dtype=torch.float)
        return x, y

    def denormalize(self, normalized_value: float) -> float:
        """Convert normalized value back to original scale."""
        if self.normalize:
            return normalized_value * self.std + self.mean
        return normalized_value


class PolymerDescriptorDataset(Dataset):
    """
    Dataset for polymer property prediction using pre-computed molecular descriptors.

    This is useful when SMILES are not available but molecular descriptors
    have been pre-computed (e.g., from quantum chemistry calculations or
    other featurization methods).
    """

    def __init__(
        self,
        csv_path: str,
        property_name: str,
        descriptor_columns: List[str],
        property_column: str = "property_value",
        normalize: bool = True,
    ):
        """
        Initialize descriptor dataset.

        Args:
            csv_path: Path to CSV file with descriptors and property values
            property_name: Name of the property (for metadata)
            descriptor_columns: List of column names containing descriptors
            property_column: Name of column containing property values
            normalize: Whether to normalize property values
        """
        self.csv_path = Path(csv_path)
        self.property_name = property_name
        self.descriptor_columns = descriptor_columns
        self.property_column = property_column

        # Load data
        self.df = pd.read_csv(csv_path)

        # Check columns exist
        missing_cols = set(descriptor_columns + [property_column]) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in {csv_path}: {missing_cols}")

        # Remove rows with missing values
        cols_to_check = descriptor_columns + [property_column]
        self.df = self.df.dropna(subset=cols_to_check)

        # Normalization
        self.mean = self.df[property_column].mean()
        self.std = self.df[property_column].std()
        self.normalize = normalize

        # Prepare descriptors
        self._prepare_descriptors()

        print(f"Loaded {len(self)} samples from {self.csv_path.name}")
        print(f"Descriptor dimension: {len(descriptor_columns)}")
        if normalize:
            print(f"Property stats: mean={self.mean:.4f}, std={self.std:.4f}")

    def _prepare_descriptors(self):
        """Extract and prepare descriptor arrays."""
        # Extract descriptor features
        self.descriptors = self.df[self.descriptor_columns].values.astype(np.float32)

        # Extract and normalize property values
        prop_values = self.df[self.property_column].values

        if self.normalize:
            self.properties = ((prop_values - self.mean) / self.std).astype(np.float32)
        else:
            self.properties = prop_values.astype(np.float32)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.descriptors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single descriptor vector and property value."""
        x = torch.from_numpy(self.descriptors[idx])
        y = torch.tensor([self.properties[idx]], dtype=torch.float)
        return x, y

    def denormalize(self, normalized_value: float) -> float:
        """Convert normalized value back to original scale."""
        if self.normalize:
            return normalized_value * self.std + self.mean
        return normalized_value

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            'property': self.property_name,
            'n_samples': len(self),
            'n_descriptors': len(self.descriptor_columns),
            'mean': self.mean,
            'std': self.std,
            'min': float(self.df[self.property_column].min()),
            'max': float(self.df[self.property_column].max()),
        }


def collate_polymer_graphs(batch: List[Data]) -> Batch:
    """
    Collate function for PyTorch DataLoader with graph data.

    Args:
        batch: List of PyTorch Geometric Data objects

    Returns:
        Batched graphs
    """
    return Batch.from_data_list(batch)


def test_dataset():
    """Test dataset loading with sample data."""
    print("Testing Polymer Dataset")
    print("=" * 60)

    # Check if sample data exists
    data_dir = Path("data/benchmark_sample")
    if not data_dir.exists():
        print("Error: Sample data not found at data/benchmark_sample/")
        print("Run: python3 scripts/download_point2_data.py")
        return

    # Test graph dataset
    print("\n1. Testing Graph Dataset (for GNNs)")
    print("-" * 60)

    train_file = data_dir / "Tg_train.csv"
    if train_file.exists():
        dataset = PolymerDataset(
            csv_path=str(train_file),
            property_name="Tg",
            normalize=True
        )

        print(f"\nDataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(f"\nSample data:")
        sample = dataset[0]
        print(f"  Graph: {sample}")
        print(f"  Nodes: {sample.x.shape}")
        print(f"  Edges: {sample.edge_index.shape}")
        print(f"  Target (normalized): {sample.y.item():.4f}")
        print(f"  Target (original): {dataset.denormalize(sample.y.item()):.4f}")

        # Test batching
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_polymer_graphs
        )

        batch = next(iter(loader))
        print(f"\nBatched data:")
        print(f"  Batch size: {batch.num_graphs}")
        print(f"  Total nodes: {batch.x.shape[0]}")
        print(f"  Total edges: {batch.edge_index.shape[1]}")
        print(f"  Targets: {batch.y.shape}")

    # Test fingerprint dataset
    print("\n\n2. Testing Fingerprint Dataset (for traditional ML)")
    print("-" * 60)

    if train_file.exists():
        fp_dataset = PolymerFingerprintDataset(
            csv_path=str(train_file),
            property_name="Tg",
            normalize=True
        )

        print(f"\nDataset size: {len(fp_dataset)}")
        print(f"Fingerprint dimension: {fp_dataset.fingerprints.shape[1]}")

        x, y = fp_dataset[0]
        print(f"\nSample:")
        print(f"  Fingerprint: {x.shape}, {x.sum().item():.0f} bits set")
        print(f"  Target: {y.item():.4f}")

    print("\n" + "=" * 60)
    print("✓ Dataset test complete!")


if __name__ == "__main__":
    test_dataset()
