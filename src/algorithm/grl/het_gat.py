"""异构图注意力编码器 (HAT)

对应论文第4章 4.3节
4阶段：类型特定投影 → 关系感知注意力聚合 → 跨类型融合 → 层次化读出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from src.problem.graph_builder import HeteroGraphData


class RelationAttentionLayer(nn.Module):
    """单关系类型的注意力聚合层"""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        attn_in = 2 * self.head_dim + (edge_dim if edge_dim > 0 else 0)
        self.attn = nn.Parameter(torch.zeros(num_heads, attn_in))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

        self.edge_proj = nn.Linear(edge_dim, edge_dim) if edge_dim > 0 else None
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_src: 源节点特征 (N_src, in_dim)
            x_dst: 目标节点特征 (N_dst, in_dim)
            edge_index: 边索引 (2, E)，[0]为源,[1]为目标
            edge_attr: 边特征 (E, edge_dim) 或 None

        Returns:
            更新后的目标节点特征 (N_dst, out_dim)
        """
        if edge_index.size(1) == 0:
            return torch.zeros(x_dst.size(0), self.head_dim * self.num_heads,
                               device=x_dst.device)

        # 线性变换
        h_src = self.W(x_src)  # (N_src, out_dim)
        h_dst = self.W(x_dst)  # (N_dst, out_dim)

        # reshape为多头 (N, num_heads, head_dim)
        h_src_heads = h_src.view(-1, self.num_heads, self.head_dim)
        h_dst_heads = h_dst.view(-1, self.num_heads, self.head_dim)

        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        # 边源和目标的特征
        h_src_e = h_src_heads[src_idx]  # (E, num_heads, head_dim)
        h_dst_e = h_dst_heads[dst_idx]  # (E, num_heads, head_dim)

        # 注意力系数
        if edge_attr is not None and self.edge_proj is not None:
            e_feat = self.edge_proj(edge_attr)  # (E, edge_dim)
            e_feat = e_feat.unsqueeze(1).expand(-1, self.num_heads, -1)  # (E, H, edge_dim)
            attn_input = torch.cat([h_src_e, h_dst_e, e_feat], dim=-1)  # (E, H, 2d+e)
        else:
            attn_input = torch.cat([h_src_e, h_dst_e], dim=-1)  # (E, H, 2d)

        # (E, H)
        e = self.leaky_relu((attn_input * self.attn.unsqueeze(0)).sum(dim=-1))

        # Softmax归一化（按目标节点分组）
        alpha = _edge_softmax(e, dst_idx, x_dst.size(0))  # (E, H)
        alpha = self.dropout(alpha)

        # 加权聚合
        msg = h_src_e * alpha.unsqueeze(-1)  # (E, H, head_dim)

        # scatter到目标节点
        out = torch.zeros(x_dst.size(0), self.num_heads, self.head_dim,
                          device=x_dst.device)
        dst_idx_expanded = dst_idx.unsqueeze(1).unsqueeze(2).expand_as(msg)
        out.scatter_add_(0, dst_idx_expanded, msg)

        # reshape回 (N_dst, out_dim)
        out = out.view(-1, self.num_heads * self.head_dim)
        return out


def _edge_softmax(e: torch.Tensor, dst_idx: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
    """对每个目标节点的入边做softmax

    Args:
        e: 注意力分数 (E, H)
        dst_idx: 目标节点索引 (E,)
        num_nodes: 目标节点总数
    """
    e_max = torch.zeros(num_nodes, e.size(1), device=e.device)
    e_max.scatter_reduce_(0, dst_idx.unsqueeze(1).expand_as(e), e, reduce='amax')
    e_shifted = e - e_max[dst_idx]

    exp_e = torch.exp(e_shifted)
    sum_exp = torch.zeros(num_nodes, e.size(1), device=e.device)
    sum_exp.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(exp_e), exp_e)
    sum_exp = sum_exp.clamp(min=1e-10)

    return exp_e / sum_exp[dst_idx]


class HATEncoder(nn.Module):
    """异构图注意力编码器

    对应论文第4章 4.3节 式(54)-(68)
    """

    # 节点特征维度
    OP_FEAT_DIM = 7
    MAC_FEAT_DIM = 6
    AGV_FEAT_DIM = 8

    # 边特征维度
    EDGE_DIMS = {
        'proc_o2m': 1, 'proc_m2o': 1,
        'seq': 1,
        'trans': 2,
        'prec': 0,
        'loc_a2m': 0, 'loc_m2a': 0,
    }

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 阶段1：类型特定线性投影
        self.proj_op = nn.Linear(self.OP_FEAT_DIM, hidden_dim)
        self.proj_mac = nn.Linear(self.MAC_FEAT_DIM, hidden_dim)
        self.proj_agv = nn.Linear(self.AGV_FEAT_DIM, hidden_dim)

        # 阶段2：关系感知注意力层
        # 定义哪些边连接哪些节点类型 (src_type, dst_type, edge_dim)
        self.relation_config = {
            'prec': ('op', 'op', 0),         # 工艺顺序
            'proc_o2m': ('op', 'mac', 1),    # 可加工 op->mac
            'proc_m2o': ('mac', 'op', 1),    # 可加工 mac->op
            'seq': ('op', 'op', 1),          # 机器顺序
            'trans': ('mac', 'mac', 2),      # 运输连接
            'loc_a2m': ('agv', 'mac', 0),    # AGV位置 agv->mac
            'loc_m2a': ('mac', 'agv', 0),    # AGV位置 mac->agv
        }

        self.attn_layers = nn.ModuleDict()
        for layer_idx in range(num_layers):
            for rel_name, (_, _, edge_dim) in self.relation_config.items():
                key = f"layer{layer_idx}_{rel_name}"
                self.attn_layers[key] = RelationAttentionLayer(
                    hidden_dim, hidden_dim, edge_dim, num_heads, dropout
                )

        # 多关系融合（每个节点类型可能有不同数量的入边关系）
        # op: prec, proc_m2o, seq (3种入边)
        # mac: proc_o2m, trans, loc_a2m (3种入边)
        # agv: loc_m2a (1种入边)
        self.fuse_op = nn.ModuleList([
            nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(num_layers)
        ])
        self.fuse_mac = nn.ModuleList([
            nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(num_layers)
        ])
        self.fuse_agv = nn.ModuleList([
            nn.Linear(hidden_dim * 1, hidden_dim) for _ in range(num_layers)
        ])

        self.layer_norms_op = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms_mac = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms_agv = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # 阶段3：跨类型信息融合
        self.cross_op = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cross_mac = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cross_agv = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 阶段4：层次化读出
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: HeteroGraphData) -> torch.Tensor:
        """前向传播

        Args:
            graph: HeteroGraphData对象

        Returns:
            全局图嵌入 (hidden_dim,)
        """
        # 阶段1：投影
        h_op = F.elu(self.proj_op(graph.x_op))    # (N_op, d)
        h_mac = F.elu(self.proj_mac(graph.x_mac))  # (N_mac, d)
        h_agv = F.elu(self.proj_agv(graph.x_agv))  # (N_agv, d)

        node_feats = {'op': h_op, 'mac': h_mac, 'agv': h_agv}

        # 阶段2：L层关系感知注意力聚合
        for layer_idx in range(self.num_layers):
            # 对每种关系计算消息
            messages = {ntype: [] for ntype in ['op', 'mac', 'agv']}

            for rel_name, (src_type, dst_type, _) in self.relation_config.items():
                key = f"layer{layer_idx}_{rel_name}"
                edge_idx = graph.edge_index.get(rel_name)
                edge_attr = graph.edge_attr.get(rel_name)

                if edge_idx is None or edge_idx.size(1) == 0:
                    msg = torch.zeros_like(node_feats[dst_type])
                else:
                    msg = self.attn_layers[key](
                        node_feats[src_type], node_feats[dst_type],
                        edge_idx, edge_attr
                    )
                messages[dst_type].append(msg)

            # 多关系融合 + 残差连接
            if messages['op']:
                while len(messages['op']) < 3:
                    messages['op'].append(torch.zeros_like(node_feats['op']))
                fused_op = self.fuse_op[layer_idx](torch.cat(messages['op'][:3], dim=-1))
                h_op_new = F.elu(fused_op) + node_feats['op']
                node_feats['op'] = self.layer_norms_op[layer_idx](h_op_new)

            if messages['mac']:
                while len(messages['mac']) < 3:
                    messages['mac'].append(torch.zeros_like(node_feats['mac']))
                fused_mac = self.fuse_mac[layer_idx](torch.cat(messages['mac'][:3], dim=-1))
                h_mac_new = F.elu(fused_mac) + node_feats['mac']
                node_feats['mac'] = self.layer_norms_mac[layer_idx](h_mac_new)

            if messages['agv']:
                fused_agv = self.fuse_agv[layer_idx](messages['agv'][0])
                h_agv_new = F.elu(fused_agv) + node_feats['agv']
                node_feats['agv'] = self.layer_norms_agv[layer_idx](h_agv_new)

        # 阶段3：跨类型信息融合
        g_op = node_feats['op'].mean(dim=0)    # (d,)
        g_mac = node_feats['mac'].mean(dim=0)  # (d,)
        g_agv = node_feats['agv'].mean(dim=0)  # (d,)

        h_op_final = self.cross_op(torch.cat([
            node_feats['op'],
            g_mac.unsqueeze(0).expand(node_feats['op'].size(0), -1),
            g_agv.unsqueeze(0).expand(node_feats['op'].size(0), -1)
        ], dim=-1))

        h_mac_final = self.cross_mac(torch.cat([
            node_feats['mac'],
            g_op.unsqueeze(0).expand(node_feats['mac'].size(0), -1),
            g_agv.unsqueeze(0).expand(node_feats['mac'].size(0), -1)
        ], dim=-1))

        h_agv_final = self.cross_agv(torch.cat([
            node_feats['agv'],
            g_op.unsqueeze(0).expand(node_feats['agv'].size(0), -1),
            g_mac.unsqueeze(0).expand(node_feats['agv'].size(0), -1)
        ], dim=-1))

        # 阶段4：层次化读出
        global_emb = self.readout(torch.cat([
            h_op_final.mean(dim=0),
            h_mac_final.mean(dim=0),
            h_agv_final.mean(dim=0)
        ], dim=-1))

        return global_emb

    def encode_batch(self, graphs: list) -> torch.Tensor:
        """编码一批图并取平均

        Args:
            graphs: HeteroGraphData对象列表

        Returns:
            平均全局嵌入 (hidden_dim,)
        """
        embeddings = [self.forward(g) for g in graphs]
        return torch.stack(embeddings).mean(dim=0)
