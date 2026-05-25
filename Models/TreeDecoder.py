import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNDecoder(nn.Module):
    def __init__(self,
                 n_exercise,
                 n_nodes,
                 final_tree,
                 Q,
                 d_model=64,
                 num_gcn_layers=2,
                 d_hidden=64,
                 use_residual=True,
                 activation="relu"):
        super().__init__()
        self.n_exercise = n_exercise
        self.d = d_model
        self.num_gcn_layers = num_gcn_layers
        self.use_residual = use_residual

        # 输入线性
        self.in_proj = nn.Linear(1, d_model)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU(0.2)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation {activation}")

        self.gcn_weights = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_gcn_layers)
        ])

        # MLP: 输入维度 = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )
        self.mlp_flatten = nn.Sequential(
            nn.Linear(n_nodes * d_model, 1024),
            nn.SiLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.5),

            nn.Linear(256, n_exercise),
        )

        self.attn_vec = torch.nn.Parameter(torch.randn(d_model))

        self.init_decoder(final_tree, Q)

    def init_decoder(self, final_tree, Q):
        # Q: (n_exercise, n_leaf)
        if not torch.is_tensor(Q):
            Q = torch.tensor(Q, dtype=torch.float32)

        n_exercise, n_leaf = Q.shape

        edges_src, edges_dst = [], []
        node_ids = []

        queue = [final_tree]
        while queue:
            node = queue.pop(0)
            nid = node["id"]
            if nid not in node_ids:
                node_ids.append(nid)
            for c in node.get("children", []):
                edges_src.append(nid)
                edges_dst.append(c["id"])
                queue.append(c)

        if len(node_ids) == 0:
            raise ValueError("final_tree contains no nodes")

        n_node = max(node_ids) + 1

        # node_mask: node -> leaves
        node_mask = torch.zeros(n_node, n_leaf, dtype=torch.bool).cuda()
        def collect_leaves(node):
            # returns list of leaf ids under node
            if not node.get("children", []):
                return [node["id"]]
            res = []
            for c in node["children"]:
                res += collect_leaves(c)
            return res

        queue = [final_tree]
        visited = set()
        while queue:
            node = queue.pop(0)
            nid = node["id"]
            if nid in visited:
                continue
            visited.add(nid)
            leaves = collect_leaves(node)
            for leaf in leaves:
                if 0 <= leaf < n_leaf:
                    node_mask[nid, leaf] = True
            for c in node.get("children", []):
                queue.append(c)

        srcs, dsts = [], []
        for s, t in zip(edges_src, edges_dst):
            srcs.append(s); dsts.append(t)
            srcs.append(t); dsts.append(s)
        if len(srcs) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

        # exercise_mask: (n_exercise, n_node)
        exercise_mask = (Q @ node_mask.T.float()) > 0
        exercise_mask = exercise_mask.float()

        # 注册为 buffer（device 会随 model.to() 或 .cuda() 移动）
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("node_mask", node_mask)
        self.register_buffer("exercise_mask", exercise_mask)

        # 记录 n_node / n_leaf
        self.n_node = n_node
        self.n_leaf = n_leaf

    def forward(self, theta):
        device = theta.device
        if theta.dim() == 2:
            h = self.in_proj(theta.unsqueeze(-1))  # (B, n_node, d)
        else:
            h = theta
            if h.size(-1) != self.d:
                h = h.view(-1, self.n_node, h.size(-1))
                h = self.in_proj(h.mean(dim=-1, keepdim=True))

        edge_index = self.edge_index.to(device)           # (2, E)
        exercise_mask = self.exercise_mask.to(device)    # (n_exercise, n_node)


        if edge_index.numel() == 0:
            deg = torch.ones(self.n_node, device=device)
            src = torch.empty(0, dtype=torch.long, device=device)
            dst = torch.empty(0, dtype=torch.long, device=device)
        else:
            src = edge_index[0].to(device)
            dst = edge_index[1].to(device)
            deg = torch.bincount(dst, minlength=self.n_node).float().to(device).clamp(min=1.0)

        B, N, d = h.shape

        for w in self.gcn_weights:
            if src.numel() == 0:
                h_new = torch.zeros_like(h)
            else:
                # messages: (B, E, d)
                messages = h[:, src, :]   # gather source node features for all edges

                # prepare index tensor to scatter into dst positions:
                # dst_idx shape (B, E, d) for scatter_add_ dim=1
                dst_idx = dst.view(1, -1, 1).expand(B, -1, d)  # on device

                # h_new: (B, N, d)
                h_new = torch.zeros_like(h, device=device)
                h_new = h_new.scatter_add_(1, dst_idx, messages)  # accumulate messages into dst nodes

                # normalize by in-degree
                inv_deg = (1.0 / deg).view(1, -1, 1)  # (1, N, 1)
                h_new = h_new * inv_deg

            # linear + activation
            h_new = w(h_new)
            h_new = self.act(h_new)

            # residual
            if self.use_residual and h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new

        logits = self.mlp_flatten(h.reshape(B, -1))

        return torch.sigmoid(logits)
