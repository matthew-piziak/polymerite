"""
Molecular featurization for polymer property prediction.

Converts polymer SMILES strings to PyTorch Geometric graphs with atom and bond features.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from typing import List, Optional, Tuple


class PolymerFeaturizer:
    """
    Convert polymer SMILES to molecular graphs for GNN models.

    Features:
    - Atom features: atomic number, degree, hybridization, aromaticity, etc.
    - Bond features: bond type, conjugation, ring membership
    - Handles polymer-specific SMILES notation (with * wildcards)
    """

    def __init__(
        self,
        atom_features: List[str] = None,
        bond_features: List[str] = None,
        add_self_loops: bool = False
    ):
        """
        Initialize featurizer.

        Args:
            atom_features: List of atom features to extract (default: all)
            bond_features: List of bond features to extract (default: all)
            add_self_loops: Whether to add self-loops to the graph
        """
        self.atom_features = atom_features or [
            'atomic_num', 'degree', 'formal_charge', 'hybridization',
            'is_aromatic', 'num_hs', 'is_in_ring'
        ]
        self.bond_features = bond_features or [
            'bond_type', 'is_conjugated', 'is_in_ring'
        ]
        self.add_self_loops = add_self_loops

        # Atom feature dimensions
        self.atomic_num_list = list(range(1, 119))  # H to Og
        self.degree_list = list(range(0, 11))
        self.formal_charge_list = list(range(-5, 6))
        self.hybridization_list = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ]
        self.num_hs_list = list(range(0, 9))

        # Bond feature dimensions
        self.bond_type_list = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]

    def _one_hot_encoding(self, value, allowable_set):
        """One-hot encode a value."""
        if value not in allowable_set:
            value = allowable_set[-1]  # Use last as "other"
        return [int(value == s) for s in allowable_set]

    def _get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """Extract features for a single atom."""
        features = []

        if 'atomic_num' in self.atom_features:
            features.extend(self._one_hot_encoding(
                atom.GetAtomicNum(), self.atomic_num_list
            ))

        if 'degree' in self.atom_features:
            features.extend(self._one_hot_encoding(
                atom.GetTotalDegree(), self.degree_list
            ))

        if 'formal_charge' in self.atom_features:
            features.extend(self._one_hot_encoding(
                atom.GetFormalCharge(), self.formal_charge_list
            ))

        if 'hybridization' in self.atom_features:
            features.extend(self._one_hot_encoding(
                atom.GetHybridization(), self.hybridization_list
            ))

        if 'is_aromatic' in self.atom_features:
            features.append(int(atom.GetIsAromatic()))

        if 'num_hs' in self.atom_features:
            features.extend(self._one_hot_encoding(
                atom.GetTotalNumHs(), self.num_hs_list
            ))

        if 'is_in_ring' in self.atom_features:
            features.append(int(atom.IsInRing()))

        return features

    def _get_bond_features(self, bond: Chem.Bond) -> List[float]:
        """Extract features for a single bond."""
        features = []

        if 'bond_type' in self.bond_features:
            features.extend(self._one_hot_encoding(
                bond.GetBondType(), self.bond_type_list
            ))

        if 'is_conjugated' in self.bond_features:
            features.append(int(bond.GetIsConjugated()))

        if 'is_in_ring' in self.bond_features:
            features.append(int(bond.IsInRing()))

        return features

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES string to PyTorch Geometric graph.

        Args:
            smiles: Polymer SMILES string (may contain * for polymerization points)

        Returns:
            PyTorch Geometric Data object with node features (x), edge indices,
            and edge features (edge_attr), or None if invalid SMILES
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add explicit hydrogens for accurate feature extraction
        mol = Chem.AddHs(mol)

        # Extract atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self._get_atom_features(atom))

        x = torch.tensor(atom_features, dtype=torch.float)

        # Extract bond features and create edge index
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_feature = self._get_bond_features(bond)

            # Add edges in both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(bond_feature)
            edge_features.append(bond_feature)

        # Handle molecules with no bonds (single atoms)
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(edge_features[0]) if edge_features else 0), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles
        )

        return data

    def get_feature_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensionality of node and edge features.

        Returns:
            (node_feature_dim, edge_feature_dim)
        """
        # Create a dummy molecule to get feature dimensions
        dummy_smiles = "CC"  # Ethane
        data = self.smiles_to_graph(dummy_smiles)

        if data is None:
            raise ValueError("Failed to create dummy molecule")

        node_dim = data.x.shape[1]
        edge_dim = data.edge_attr.shape[1] if data.edge_attr.numel() > 0 else 0

        return node_dim, edge_dim


class MorganFingerprintFeaturizer:
    """
    Alternative featurizer using Morgan fingerprints (circular fingerprints).

    This is useful for comparison with traditional ML methods and provides
    a fixed-length feature vector instead of a graph.
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        """
        Initialize Morgan fingerprint featurizer.

        Args:
            radius: Radius for Morgan fingerprint (default: 2)
            n_bits: Number of bits in fingerprint (default: 2048)
        """
        self.radius = radius
        self.n_bits = n_bits

    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert SMILES to Morgan fingerprint.

        Args:
            smiles: Polymer SMILES string

        Returns:
            NumPy array of fingerprint bits, or None if invalid SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits
        )

        return np.array(fp)

    def get_feature_dimension(self) -> int:
        """Get fingerprint dimension."""
        return self.n_bits


def test_featurization():
    """Test featurization on example polymer SMILES."""
    print("Testing Polymer Featurization")
    print("=" * 60)

    # Test SMILES
    test_molecules = {
        "Polyethylene": "*CC(*)",
        "Polypropylene": "*CC(*)C",
        "Polystyrene": "*CC(*)c1ccccc1",
        "PET": "*OC(=O)c1ccc(C(=O)O*)cc1",
    }

    # Test graph featurizer
    print("\n1. Graph Featurization (for GNNs)")
    print("-" * 60)

    featurizer = PolymerFeaturizer()
    node_dim, edge_dim = featurizer.get_feature_dimensions()
    print(f"Feature dimensions: nodes={node_dim}, edges={edge_dim}")
    print()

    for name, smiles in test_molecules.items():
        data = featurizer.smiles_to_graph(smiles)
        if data is not None:
            print(f"{name:20s} ({smiles})")
            print(f"  Nodes: {data.x.shape[0]:3d}, Edges: {data.edge_index.shape[1]:3d}")
            print(f"  Node features: {data.x.shape}")
            print(f"  Edge features: {data.edge_attr.shape}")
        else:
            print(f"{name:20s} - Failed to parse SMILES")
        print()

    # Test fingerprint featurizer
    print("\n2. Morgan Fingerprint Featurization (for traditional ML)")
    print("-" * 60)

    fp_featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
    print(f"Fingerprint dimension: {fp_featurizer.get_feature_dimension()}")
    print()

    for name, smiles in test_molecules.items():
        fp = fp_featurizer.smiles_to_fingerprint(smiles)
        if fp is not None:
            n_bits_set = np.sum(fp)
            print(f"{name:20s}: {n_bits_set}/{len(fp)} bits set")
        else:
            print(f"{name:20s} - Failed to parse SMILES")

    print("\n" + "=" * 60)
    print("✓ Featurization test complete!")


if __name__ == "__main__":
    test_featurization()
