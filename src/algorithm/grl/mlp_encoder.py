"""MLP编码器（消融变体V7的替代品）

与HAT有相同的参数量，但不使用图结构——
将所有节点特征拼接/聚合后用MLP编码。
用于论文消融实验：证明GNN的图结构编码比MLP的扁平编码更有效。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.problem.graph_builder import HeteroGraphData


class MLPEncoder(nn.Module):
    """MLP编码器（作为HAT的消融替代品）

    将所有节点特征平均池化后拼接，通过MLP编码为全局嵌入。
    参数量与HAT接近但不利用图结构。
    """

    OP_FEAT_DIM = 7
    MAC_FEAT_DIM = 6
    AGV_FEAT_DIM = 8

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        input_dim = self.OP_FEAT_DIM + self.MAC_FEAT_DIM + self.AGV_FEAT_DIM  # 21

        layers = []
        dims = [input_dim] + [hidden_dim * 2] * num_layers + [hidden_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, graph: HeteroGraphData) -> torch.Tensor:
        """将异构图的节点特征聚合后用MLP编码

        Returns:
            全局嵌入 (hidden_dim,)
        """
        # 平均池化各类型节点特征
        op_mean = graph.x_op.mean(dim=0) if graph.x_op is not None and len(graph.x_op) > 0 \
            else torch.zeros(self.OP_FEAT_DIM)
        mac_mean = graph.x_mac.mean(dim=0) if graph.x_mac is not None and len(graph.x_mac) > 0 \
            else torch.zeros(self.MAC_FEAT_DIM)
        agv_mean = graph.x_agv.mean(dim=0) if graph.x_agv is not None and len(graph.x_agv) > 0 \
            else torch.zeros(self.AGV_FEAT_DIM)

        # 拼接 → MLP
        x = torch.cat([op_mean, mac_mean, agv_mean], dim=0)  # (21,)
        return self.mlp(x)  # (hidden_dim,)

    def encode_batch(self, graphs: list) -> torch.Tensor:
        embeddings = [self.forward(g) for g in graphs]
        return torch.stack(embeddings).mean(dim=0)
