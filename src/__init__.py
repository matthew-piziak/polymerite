"""
Polymerite: Polymer Property Prediction with Machine Learning

This package provides tools for:
- Featurizing polymer SMILES strings
- Loading and preparing datasets
- Training GNN and transformer models
- Evaluating model performance
"""

from .featurization import (
    PolymerFeaturizer,
    MorganFingerprintFeaturizer,
)

from .dataset import (
    PolymerDataset,
    PolymerFingerprintDataset,
    collate_polymer_graphs,
)

__version__ = "0.1.0"

__all__ = [
    "PolymerFeaturizer",
    "MorganFingerprintFeaturizer",
    "PolymerDataset",
    "PolymerFingerprintDataset",
    "collate_polymer_graphs",
]
