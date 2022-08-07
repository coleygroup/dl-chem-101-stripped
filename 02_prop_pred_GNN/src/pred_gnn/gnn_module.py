
import numpy as np

import torch
import torch.nn as nn

import dgl.function as fn
from dgllife.model import MPNNGNN
from dgl.backend import pytorch as dgl_F


class MoleculeGNN(nn.Module):
    

    def __init__(
            self,
            hidden_size: int,
            num_step_message_passing: int = 4,
            gnn_node_feats: int = 74,
            gnn_edge_feats: int = 4,  
            mpnn_type: str = "NNConv",
            node_feat_symbol="h",
            set_transform_layers: int = 2,
            **kwargs):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_edge_feats = gnn_edge_feats
        self.gnn_node_feats = gnn_node_feats
        self.node_feat_symbol = node_feat_symbol

        self.mpnn_type = mpnn_type
        node_out_feats = self.hidden_size
        if self.mpnn_type == "NNConv":
            
            
            
            self.gnn = MPNNGNN(
                node_in_feats=self.gnn_node_feats,
                edge_in_feats=self.gnn_edge_feats,
                node_out_feats=self.hidden_size,
                edge_hidden_feats=self.hidden_size // 4,  
                num_step_message_passing=num_step_message_passing,
            )
        elif self.mpnn_type == "GGNN":
            self.gnn = GGNN(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
            )
        else:
            raise ValueError()

        
        
        self.set_transformer = SetTransformerEncoder(
            d_model=node_out_feats,
            n_heads=4,
            d_head=node_out_feats // 4,
            d_ff=hidden_size,
            n_layers=set_transform_layers,
        )

    def forward(self, g):
        
        node_feats = g.ndata[self.node_feat_symbol]
        device = g.device
        if self.mpnn_type == "NNConv":
            edge_feats = g.edata.get(
                "e", torch.empty(0, self.gnn_edge_feats, device=device))
            output = self.gnn(g, node_feats, edge_feats)
        elif self.mpnn_type == "GGNN":
            
            output = self.gnn(g, self.node_feat_symbol, "e")
        else:
            raise NotImplementedError()

        output = self.set_transformer(g, output)
        return output


class GGNN(nn.Module):

    def __init__(self,
                 hidden_size=64,
                 edge_feats=4,
                 node_feats=74,
                 num_step_message_passing=4,
                 **kwargs):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_step_message_passing = num_step_message_passing
        self.edge_feats = edge_feats
        self.node_feats = node_feats
        self.input_project = nn.Linear(self.node_feats, self.hidden_size)

        
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.edge_transform_weights = torch.nn.Parameter(
            torch.randn(self.edge_feats, self.hidden_size, self.hidden_size))
        self.edge_transform_bias = torch.nn.Parameter(
            torch.randn(self.edge_feats, 1))

        
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.tan_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )

    def message_pass(self, edges):
        
        src_feat = edges.src["_h"]

        
        messages = (torch.einsum("nio,ni->no", edges.data["w"], src_feat) +
                    edges.data["b"])
        messages = nn.functional.relu(messages)
        return {"m": messages}

    def forward(self, graph, nfeat_name="h", efeat_name="e"):
        
        ndata = graph.ndata[nfeat_name]
        with graph.local_scope():

            
            h_init = self.input_project(ndata)
            graph.ndata.update({"_h": h_init})

            
            efeats = graph.edata[efeat_name].argmax(1)
            for layer in range(self.num_step_message_passing):
                h_in = graph.ndata["_h"]
                edge_transforms = self.edge_transform_weights[efeats]
                edge_biases = self.edge_transform_bias[efeats]
                graph.edata.update({"w": edge_transforms, "b": edge_biases})
                graph.update_all(self.message_pass, fn.sum("m", "_h"))
                h_out = graph.ndata["_h"]
                h_out = self.gru(h_out, h_in)
                graph.ndata.update({"_h": h_out})

            
            
            node_level_output = torch.cat([h_init, h_out], -1)
            
            
            output_val = self.tan_mlp(node_level_output)
        return output_val






class MultiHeadAttention(nn.Module):
    

    def __init__(self,
                 d_model,
                 num_heads,
                 d_head,
                 d_ff,
                 dropouth=0.0,
                 dropouta=0.0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.proj_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model),
        )
        self.droph = nn.Dropout(dropouth)
        self.dropa = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_inter = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def self_attention(self, x, mem, lengths_x, lengths_mem):
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        max_len_mem = max(lengths_mem)
        device = x.device
        lengths_x = torch.tensor(lengths_x, dtype=torch.int64, device=device)
        lengths_mem = torch.tensor(lengths_mem,
                                   dtype=torch.int64,
                                   device=device)

        queries = self.proj_q(x).view(-1, self.num_heads, self.d_head)
        keys = self.proj_k(mem).view(-1, self.num_heads, self.d_head)
        values = self.proj_v(mem).view(-1, self.num_heads, self.d_head)

        
        queries = dgl_F.pad_packed_tensor(queries, lengths_x, 0)
        keys = dgl_F.pad_packed_tensor(keys, lengths_mem, 0)
        values = dgl_F.pad_packed_tensor(values, lengths_mem, 0)

        
        e = torch.einsum("bxhd,byhd->bhxy", queries, keys)
        
        e = e / np.sqrt(self.d_head)

        
        mask = _gen_mask(lengths_x, lengths_mem, max_len_x, max_len_mem)
        e = e.masked_fill(mask == 0, -float("inf"))

        
        alpha = torch.softmax(e, dim=-1)
        
        
        alpha = alpha.masked_fill(mask == 0, 0.0)

        
        out = torch.einsum("bhxy,byhd->bxhd", alpha, values)
        
        out = self.proj_o(out.contiguous().view(batch_size, max_len_x,
                                                self.num_heads * self.d_head))
        
        out = dgl_F.pack_padded_tensor(out, lengths_x)
        return out

    def forward(self, x, mem, lengths_x, lengths_mem):
        

        

        
        x = x + self.self_attention(self.norm_in(x), mem, lengths_x,
                                    lengths_mem)

        
        x = x + self.ffn(self.norm_inter(x))

        
        

        
        
        return x


class SetAttentionBlock(nn.Module):
    

    def __init__(self,
                 d_model,
                 num_heads,
                 d_head,
                 d_ff,
                 dropouth=0.0,
                 dropouta=0.0):
        super(SetAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model,
                                      num_heads,
                                      d_head,
                                      d_ff,
                                      dropouth=dropouth,
                                      dropouta=dropouta)

    def forward(self, feat, lengths):
        
        return self.mha(feat, feat, lengths, lengths)


class SetTransformerEncoder(nn.Module):
    

    def __init__(
        self,
        d_model,
        n_heads,
        d_head,
        d_ff,
        n_layers=1,
        block_type="sab",
        m=None,
        dropouth=0.0,
        dropouta=0.0,
    ):
        super(SetTransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.m = m
        layers = []
        if block_type == "isab" and m is None:
            raise KeyError(
                "The number of inducing points is not specified in ISAB block."
            )

        for _ in range(n_layers):
            if block_type == "sab":
                layers.append(
                    SetAttentionBlock(
                        d_model,
                        n_heads,
                        d_head,
                        d_ff,
                        dropouth=dropouth,
                        dropouta=dropouta,
                    ))
            elif block_type == "isab":
                
                
                
                raise NotImplementedError()
            else:
                raise KeyError(
                    "Unrecognized block type {}: we only support sab/isab")

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        
        lengths = graph.batch_num_nodes()
        for layer in self.layers:
            feat = layer(feat, lengths)
        return feat


def _gen_mask(lengths_x, lengths_y, max_len_x, max_len_y):
    
    device = lengths_x.device
    
    x_mask = torch.arange(max_len_x,
                          device=device).unsqueeze(0) < lengths_x.unsqueeze(1)
    
    y_mask = torch.arange(max_len_y,
                          device=device).unsqueeze(0) < lengths_y.unsqueeze(1)
    
    mask = (x_mask.unsqueeze(-1) & y_mask.unsqueeze(-2)).unsqueeze(1)
    return mask


def pad_packed_tensor(input, lengths, value):
    
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    
    index = torch.ones(len(input), dtype=torch.int64, device=device)

    
    row_shifts = torch.cumsum(max_len - lengths, 0)

    
    
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])

    
    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0]:] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])
