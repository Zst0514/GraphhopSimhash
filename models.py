import torch
import torch.nn.functional as F

from .paths import ensure_repo_paths

ensure_repo_paths()
from gp.nn.layer.pyg import RGATEdgeConv  # noqa: E402

class GNN_LLM_Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder_proj = torch.nn.Linear(in_channels, hidden_channels)
        self.encoder = self.encoder_proj
        self.conv1 = RGATEdgeConv(hidden_channels, hidden_channels, num_relations=1, heads=8)
        self.conv2 = RGATEdgeConv(hidden_channels, hidden_channels, num_relations=1, heads=8)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward_gnn_only(self, x_emb, edge_index, edge_type=None, edge_attr=None):
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x_emb.device)
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), x_emb.size(1), device=x_emb.device)

        x = self.conv1(x_emb, edge_attr, edge_index, edge_type).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_attr, edge_index, edge_type)
        return self.classifier(x)
