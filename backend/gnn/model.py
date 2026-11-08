import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from backend.gnn.graph_builder import FEATURE_WIDTHS, RELATIONS


class DriverHGT(nn.Module):
    """Two-hop, relation-aware message passing with a variant-level logit."""

    def __init__(self, hidden: int = 32, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        metadata = (list(FEATURE_WIDTHS), list(RELATIONS))
        self.projections = nn.ModuleDict({name: nn.Linear(width, hidden) for name, width in FEATURE_WIDTHS.items()})
        self.layers = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(2)])
        self.norms = nn.ModuleList([
            nn.ModuleDict({name: nn.LayerNorm(hidden) for name in FEATURE_WIDTHS}) for _ in self.layers
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, graph: HeteroData) -> Tensor:
        features = {name: F.gelu(self.projections[name](x)) for name, x in graph.x_dict.items()}
        edges = graph.edge_index_dict
        has_messages = {}
        for name, x in features.items():
            incoming = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            for (_, _, target), index in edges.items():
                if target == name:
                    incoming[index[1]] = True
            has_messages[name] = incoming.unsqueeze(1)

        if any(index.numel() for index in edges.values()):
            for layer, norms in zip(self.layers, self.norms, strict=True):
                updates = layer(features, edges)
                for name, update in updates.items():
                    if update is not None:
                        residual = norms[name](features[name] + self.dropout(update))
                        features[name] = torch.where(has_messages[name], residual, features[name])
        return self.head(features["variant"]).squeeze(-1)
