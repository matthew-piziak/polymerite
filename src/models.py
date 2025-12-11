"""
Graph Neural Network models for polymer property prediction.

Implements MPNN (Message Passing Neural Network) and other GNN architectures
for predicting polymer properties from molecular graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch


class MPNNConv(MessagePassing):
    """
    Message Passing Neural Network Convolution Layer.

    Implements message passing between atoms (nodes) in a molecular graph,
    with edge features to encode bond information.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        aggr: str = "add"
    ):
        """
        Initialize MPNN convolution layer.

        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Dimension of hidden representations
            aggr: Aggregation method ('add', 'mean', 'max')
        """
        super().__init__(aggr=aggr)

        # Message function: combines source node and edge features
        self.message_nn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update function: combines node features with aggregated messages
        self.update_nn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """
        Construct messages from neighboring nodes.

        Args:
            x_j: Features of source nodes [num_edges, node_dim]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Messages [num_edges, hidden_dim]
        """
        # Concatenate node and edge features
        combined = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_nn(combined)

    def update(self, aggr_out, x):
        """
        Update node features with aggregated messages.

        Args:
            aggr_out: Aggregated messages [num_nodes, hidden_dim]
            x: Current node features [num_nodes, node_dim]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Concatenate current features with aggregated messages
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_nn(combined)


class MPNN(nn.Module):
    """
    Message Passing Neural Network for polymer property prediction.

    Architecture:
    1. Node embedding layer
    2. Multiple MPNN convolution layers with residual connections
    3. Global pooling to get graph-level representation
    4. MLP for property prediction
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        output_dim: int = 1
    ):
        """
        Initialize MPNN model.

        Args:
            node_dim: Input node feature dimension
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden dimension for message passing
            num_layers: Number of message passing layers
            dropout: Dropout rate
            pooling: Global pooling method ('mean', 'add', 'max')
            output_dim: Output dimension (1 for regression)
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # Node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # MPNN layers
        self.conv_layers = nn.ModuleList([
            MPNNConv(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Global pooling
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "add":
            self.pool = global_add_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: Batched PyTorch Geometric data with:
                - x: Node features [total_nodes, node_dim]
                - edge_index: Edge indices [2, total_edges]
                - edge_attr: Edge features [total_edges, edge_dim]
                - batch: Batch assignment [total_nodes]

        Returns:
            Predictions [batch_size, output_dim]
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial node embedding
        h = self.node_embedding(x)

        # Message passing layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connection (skip first layer)
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # Global pooling to get graph-level representation
        h_graph = self.pool(h, batch)

        # Predict property
        out = self.predictor(h_graph)

        return out

    def get_embeddings(self, data: Batch) -> torch.Tensor:
        """
        Get graph-level embeddings (for visualization or transfer learning).

        Args:
            data: Batched PyTorch Geometric data

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        h = self.node_embedding(x)

        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)

        h_graph = self.pool(h, batch)

        return h_graph


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for polymer property prediction from fingerprints.

    This serves as a baseline model for comparison with graph neural networks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Input feature dimension (fingerprint size)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
            output_dim: Output dimension (1 for regression)
        """
        super().__init__()

        layers = []

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predictions [batch_size, output_dim]
        """
        return self.model(x)


def test_models():
    """Test model instantiation and forward pass."""
    print("Testing GNN Models")
    print("=" * 60)

    # Create dummy data
    batch_size = 4
    num_nodes_per_graph = [10, 15, 8, 12]
    node_dim = 158  # From featurizer
    edge_dim = 6    # From featurizer
    hidden_dim = 128

    # Create batched graph data
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []

    for i, num_nodes in enumerate(num_nodes_per_graph):
        # Random node features
        x = torch.randn(num_nodes, node_dim)
        x_list.append(x)

        # Random edges (fully connected for simplicity)
        num_edges = num_nodes * (num_nodes - 1)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_index_list.append(edge_index)

        # Random edge features
        edge_attr = torch.randn(num_edges, edge_dim)
        edge_attr_list.append(edge_attr)

        # Batch assignment
        batch_list.append(torch.full((num_nodes,), i, dtype=torch.long))

    # Concatenate into batch
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    batch = torch.cat(batch_list, dim=0)
    y = torch.randn(batch_size, 1)

    # Create batch object
    data = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)

    print(f"Batch data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total nodes: {x.shape[0]}")
    print(f"  Total edges: {edge_index.shape[1]}")
    print(f"  Node features: {x.shape}")
    print(f"  Edge features: {edge_attr.shape}")
    print()

    # Test MPNN
    print("1. Testing MPNN")
    print("-" * 60)

    model = MPNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    out = model(data)
    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output values: {out.squeeze()}")

    # Test embeddings
    embeddings = model.get_embeddings(data)
    print(f"Embeddings shape: {embeddings.shape}")
    print()

    # Test MLP
    print("2. Testing SimpleMLP")
    print("-" * 60)

    fp_dim = 2048
    mlp = SimpleMLP(
        input_dim=fp_dim,
        hidden_dim=512,
        num_layers=3,
        dropout=0.2,
    )

    print(f"Model parameters: {sum(p.numel() for p in mlp.parameters()):,}")

    # Forward pass
    x_fp = torch.randn(batch_size, fp_dim)
    out_mlp = mlp(x_fp)
    print(f"Input shape: {x_fp.shape}")
    print(f"Output shape: {out_mlp.shape}")
    print(f"Output values: {out_mlp.squeeze()}")

    print("\n" + "=" * 60)
    print("✓ Model tests complete!")


if __name__ == "__main__":
    test_models()
