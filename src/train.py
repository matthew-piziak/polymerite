"""
Training script for polymer property prediction models.

Supports both MPNN (graph) and MLP (fingerprint) models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from typing import Dict, Tuple

from .featurization import PolymerFeaturizer, MorganFingerprintFeaturizer
from .dataset import (
    PolymerDataset,
    PolymerFingerprintDataset,
    PolymerDescriptorDataset,
    collate_polymer_graphs,
)
from .models import MPNN, SimpleMLP


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dictionary with RMSE, MAE, and R²
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
    }


class Trainer:
    """Training manager for polymer property prediction models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            device: Device to train on ('cpu', 'cuda', or 'cuda:0')
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, loader: DataLoader, is_graph_model: bool = True) -> float:
        """
        Train for one epoch.

        Args:
            loader: Training data loader
            is_graph_model: Whether model expects graph data (MPNN) or tensors (MLP)

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in loader:
            if is_graph_model:
                # Graph model (MPNN)
                batch = batch.to(self.device)
                predictions = self.model(batch).squeeze()  # [batch, 1] -> [batch]
                targets = batch.y
            else:
                # Fingerprint model (MLP)
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                predictions = self.model(x).squeeze()  # [batch, 1] -> [batch]
                targets = y.squeeze()  # Ensure targets are also flat

            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        is_graph_model: bool = True,
        denormalize_fn=None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on a dataset.

        Args:
            loader: Evaluation data loader
            is_graph_model: Whether model expects graph data
            denormalize_fn: Function to convert normalized predictions back to original scale

        Returns:
            (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_predictions = []
        all_targets = []

        for batch in loader:
            if is_graph_model:
                batch = batch.to(self.device)
                predictions = self.model(batch).squeeze()  # [batch, 1] -> [batch]
                targets = batch.y
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                predictions = self.model(x).squeeze()  # [batch, 1] -> [batch]
                targets = y.squeeze()  # Ensure targets are also flat

            loss = self.criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1

            # Store for metrics calculation
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

        avg_loss = total_loss / num_batches

        # Convert to numpy arrays
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)

        # Denormalize if function provided
        if denormalize_fn is not None:
            predictions_np = np.array([denormalize_fn(p) for p in predictions_np])
            targets_np = np.array([denormalize_fn(t) for t in targets_np])

        # Compute metrics on denormalized values
        metrics = compute_metrics(predictions_np, targets_np)

        return avg_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 10,
        is_graph_model: bool = True,
        denormalize_fn=None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            is_graph_model: Whether model expects graph data
            denormalize_fn: Function to denormalize predictions
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        epochs_without_improvement = 0

        if verbose:
            print(f"Training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print()

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, is_graph_model)

            # Validation
            val_loss, val_metrics = self.evaluate(
                val_loader,
                is_graph_model,
                denormalize_fn
            )
            self.val_losses.append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val RMSE: {val_metrics['rmse']:.4f} | "
                      f"Val R²: {val_metrics['r2']:.4f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }


def train_mpnn(
    property_name: str,
    train_csv: str,
    test_csv: str,
    output_dir: str = "models/checkpoints",
    hidden_dim: int = 128,
    num_layers: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    patience: int = 10,
    device: str = None,
):
    """
    Train MPNN model on polymer property data.

    Args:
        property_name: Name of property to predict (e.g., 'Tg')
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        output_dir: Directory to save model checkpoints
        hidden_dim: Hidden dimension for MPNN
        num_layers: Number of message passing layers
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on (default: auto-detect)
    """
    print("=" * 70)
    print(f"Training MPNN for {property_name}")
    print("=" * 70)
    print()

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    featurizer = PolymerFeaturizer()

    train_dataset = PolymerDataset(
        csv_path=train_csv,
        property_name=property_name,
        featurizer=featurizer,
        normalize=True,
    )

    test_dataset = PolymerDataset(
        csv_path=test_csv,
        property_name=property_name,
        featurizer=featurizer,
        normalize=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_polymer_graphs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_polymer_graphs,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print()

    # Create model
    node_dim, edge_dim = featurizer.get_feature_dimensions()
    model = MPNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
    )

    # Train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        patience=patience,
        is_graph_model=True,
        denormalize_fn=train_dataset.denormalize,
        verbose=True,
    )

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    test_loss, test_metrics = trainer.evaluate(
        test_loader,
        is_graph_model=True,
        denormalize_fn=train_dataset.denormalize,
    )

    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")

    # Save model and results
    model_path = output_dir / f"mpnn_{property_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
        },
        'dataset_stats': {
            'mean': train_dataset.mean,
            'std': train_dataset.std,
        },
        'test_metrics': test_metrics,
    }, model_path)

    print(f"\nModel saved to: {model_path}")

    # Save metrics
    results = {
        'property': property_name,
        'model': 'MPNN',
        'test_metrics': test_metrics,
        'history': history,
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        }
    }

    results_path = output_dir / f"mpnn_{property_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print()

    return model, test_metrics


def train_mlp(
    property_name: str,
    train_csv: str,
    test_csv: str,
    output_dir: str = "models/checkpoints",
    hidden_dim: int = 512,
    num_layers: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    patience: int = 10,
    device: str = None,
):
    """
    Train SimpleMLP model on polymer property data using Morgan fingerprints.

    Args:
        property_name: Name of property to predict
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        output_dir: Directory to save model checkpoints
        hidden_dim: Hidden dimension for MLP
        num_layers: Number of hidden layers
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on (default: auto-detect)
    """
    print("=" * 70)
    print(f"Training SimpleMLP for {property_name}")
    print("=" * 70)
    print()

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)

    train_dataset = PolymerFingerprintDataset(
        csv_path=train_csv,
        property_name=property_name,
        featurizer=featurizer,
        normalize=True,
    )

    test_dataset = PolymerFingerprintDataset(
        csv_path=test_csv,
        property_name=property_name,
        featurizer=featurizer,
        normalize=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print()

    # Create model
    input_dim = featurizer.get_feature_dimension()
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
    )

    # Train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        patience=patience,
        is_graph_model=False,
        denormalize_fn=train_dataset.denormalize,
        verbose=True,
    )

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    test_loss, test_metrics = trainer.evaluate(
        test_loader,
        is_graph_model=False,
        denormalize_fn=train_dataset.denormalize,
    )

    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")

    # Save model and results
    model_path = output_dir / f"mlp_{property_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
        },
        'dataset_stats': {
            'mean': train_dataset.mean,
            'std': train_dataset.std,
        },
        'test_metrics': test_metrics,
    }, model_path)

    print(f"\nModel saved to: {model_path}")

    # Save metrics
    results = {
        'property': property_name,
        'model': 'SimpleMLP',
        'representation': 'Morgan Fingerprints (2048-bit)',
        'test_metrics': test_metrics,
        'history': history,
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        }
    }

    results_path = output_dir / f"mlp_{property_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print()

    return model, test_metrics


def train_mlp_descriptors(
    property_name: str,
    train_csv: str,
    test_csv: str,
    descriptor_columns: list,
    output_dir: str = "models/checkpoints",
    hidden_dim: int = 512,
    num_layers: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    patience: int = 10,
    device: str = None,
):
    """
    Train SimpleMLP model using pre-computed molecular descriptors.

    This function is specifically for datasets that provide molecular descriptors
    instead of SMILES strings (e.g., PUE643 polyurethane elastomer dataset).

    Args:
        property_name: Name of property to predict
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        descriptor_columns: List of column names containing descriptor features
        output_dir: Directory to save model checkpoints
        hidden_dim: Hidden dimension for MLP
        num_layers: Number of hidden layers
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on (default: auto-detect)
    """
    print("=" * 70)
    print(f"Training SimpleMLP (Descriptors) for {property_name}")
    print("=" * 70)
    print()

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")

    train_dataset = PolymerDescriptorDataset(
        csv_path=train_csv,
        property_name=property_name,
        descriptor_columns=descriptor_columns,
        normalize=True,
    )

    test_dataset = PolymerDescriptorDataset(
        csv_path=test_csv,
        property_name=property_name,
        descriptor_columns=descriptor_columns,
        normalize=True,
    )

    # Use training set statistics for normalization
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset:  {len(test_dataset)} samples")
    print(f"Descriptor dimension: {len(descriptor_columns)}")
    print()

    # Create model
    input_dim = len(descriptor_columns)
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
    )

    # Train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        patience=patience,
        is_graph_model=False,
        denormalize_fn=train_dataset.denormalize,
        verbose=True,
    )

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    test_loss, test_metrics = trainer.evaluate(
        test_loader,
        is_graph_model=False,
        denormalize_fn=train_dataset.denormalize,
    )

    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")

    # Save model
    model_path = output_dir / f"mlp_descriptors_{property_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
        },
        'descriptor_columns': descriptor_columns,
        'dataset_stats': {
            'mean': train_dataset.mean,
            'std': train_dataset.std,
        },
        'test_metrics': test_metrics,
    }, model_path)

    print(f"\nModel saved to: {model_path}")

    # Save metrics
    results = {
        'property': property_name,
        'model': 'SimpleMLP',
        'representation': f'Molecular Descriptors ({len(descriptor_columns)}D)',
        'test_metrics': test_metrics,
        'history': history,
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        }
    }

    results_path = output_dir / f"mlp_descriptors_{property_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print()

    return model, test_metrics


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train polymer property prediction model')
    parser.add_argument('--property', type=str, required=True,
                        help='Property to predict (e.g., Tg, density, TC)')
    parser.add_argument('--train-csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--model', type=str, default='mpnn',
                        choices=['mpnn', 'mlp'],
                        help='Model architecture')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda)')
    parser.add_argument('--output-dir', type=str, default='models/checkpoints',
                        help='Output directory for models')

    args = parser.parse_args()

    if args.model == 'mpnn':
        train_mpnn(
            property_name=args.property,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
            device=args.device,
        )
    elif args.model == 'mlp':
        train_mlp(
            property_name=args.property,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
            device=args.device,
        )
    else:
        raise NotImplementedError(f"Model {args.model} not yet implemented")


if __name__ == "__main__":
    main()
