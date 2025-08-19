import math
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv


class deep_gcn(nn.Module):
    def __init__(self, input_dim, hidden, classes, num_layer, dropout, activation):
        super(deep_gcn, self).__init__()
        self.num_layer = num_layer
        self.layer = nn.ModuleList()
        self.drop = dropout
        self.acti = activation

        for i in range(num_layer):
            in_feat = input_dim if i == 0 else hidden
            out_feat = hidden if i < num_layer-1 else classes
            self.layer.append(GraphConv(in_feats=in_feat, out_feats=out_feat, allow_zero_in_degree=True, bias=False))

    def forward(self, g_list, features):
        g = g_list
        x = features
        h = F.dropout(x, self.drop[0], self.training)

        for j in range(self.num_layer):
            h = self.layer[j](g, h)
            if j < self.num_layer - 1:
                if self.acti:
                    h = F.relu(h)
                h = F.dropout(h, self.drop[1], self.training)

        return h


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout, activation, bn):
        super(MLP, self).__init__()
        self.nl = num_layers
        self.drop = dropout
        self.acti = activation
        self.bn = bn

        self.bnlayer = nn.ModuleList()
        if bn:
            for i in range(num_layers):
                if i == 0:
                    self.bnlayer.append(nn.BatchNorm1d(input_dim))
                if 0 < i < num_layers:
                    self.bnlayer.append(nn.BatchNorm1d(hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            if 0 < i < num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.init_parameter()

    def init_parameter(self):
        for i in range(self.nl):
            stdv = 1. / math.sqrt(self.layers[i].weight.size(1))
            self.layers[i].weight.data.normal_(-stdv, stdv)

    def forward(self, x):
        if self.bn:
            x = self.bnlayer[0](x)
        h = F.dropout(x, self.drop[0], self.training)
        for i in range(self.nl):
            h = self.layers[i](h)
            if i < self.nl - 1:
                if self.acti:
                    h = F.relu(h)
                    # h = F.leaky_relu(h)
                if self.bn and i+1 < self.nl:
                    h = self.bnlayer[i+1](h)
                h = F.dropout(h, self.drop[1], self.training)

        return h


class GCNBase(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers):
        super(GCNBase, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, out_dim, num_layers, [0,0], True, False)


    def forward(self, adj, x):
        x = torch.mm(adj, x)
        x = torch.mm(adj, x)

        h = self.mlp(x)

        return h


class TFE_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout, activation, hop, combine):
        super(TFE_GNN, self).__init__()
        self.nl = num_layers
        self.drop = dropout
        self.acti = activation
        # hop can be a list of per-band Ks
        self.hops = hop if isinstance(hop, (list, tuple)) else [hop]
        self.combine = combine

        # Per-band polynomial coefficients (length K_i + 1 each)
        self.adaptives = nn.ParameterList([nn.Parameter(torch.Tensor(h + 1)) for h in self.hops])

        # Back-compat attributes (used by older training code)
        self.adaptive_lp = self.adaptives[0]
        self.adaptive = self.adaptives[-1]

        # Fusion coefficients per band (used when sum/concat)
        if self.combine in ('sum', 'con'):
            self.ense_coe = nn.Parameter(torch.Tensor(len(self.adaptives)))

        # Linear stack: first layer depends on concat vs sum
        self.nlayers_input_dim = input_dim * (len(self.adaptives) if self.combine == 'con' else 1)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(self.nlayers_input_dim, hidden_dim, bias=False))
            elif i < num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            else:
                self.layers.append(nn.Linear(hidden_dim, out_dim, bias=False))

        self.init_parameter()

    def init_parameter(self):
        # Initialize fusion and polynomial coefficients
        if hasattr(self, 'ense_coe'):
            self.ense_coe.data.fill_(1.0)
        for p in self.adaptives:
            p.data.fill_(0.5)

        # Initialize linear layers
        for i in range(self.nl):
            stdv = 1. / math.sqrt(self.layers[i].weight.size(1))
            self.layers[i].weight.data.normal_(-stdv, stdv)

    def mix_prop(self, adj, x, coe):
        # Polynomial propagation: sum_k coe[k] * (A^k x)
        x0 = x.clone()
        xx = x.clone() * coe[0]
        for i in range(1, len(coe)):
            x0 = torch.mm(adj, x0)
            xx = xx + coe[i] * x0
        return xx

    def forward(self, adj_hp, adj_lp, h0):
        # Accept either a list of adjs or the traditional (adj_hp, adj_lp)
        if isinstance(adj_hp, (list, tuple)):
            adjs = list(adj_hp)
            adaptives = list(self.adaptives)
            # If there are more adaptives than adjs (or vice versa), match the shorter
            bands = min(len(adaptives), len(adjs))
            adaptives = adaptives[:bands]
            adjs = adjs[:bands]
        else:
            # Back-compat: two-band path
            adjs = [adj_lp, adj_hp]
            adaptives = [self.adaptive_lp, self.adaptive]

        if self.drop[0] > 0:
            h0 = F.dropout(h0, self.drop[0], self.training)

        # Per-band propagation
        h_bands = [self.mix_prop(A, h0, coe) for A, coe in zip(adjs, adaptives)]

        # Fusion
        if self.combine == 'sum':
            if hasattr(self, 'ense_coe'):
                h = sum(self.ense_coe[i] * hbi for i, hbi in enumerate(h_bands))
            else:
                h = sum(h_bands)
        elif self.combine == 'con':
            if hasattr(self, 'ense_coe'):
                h = torch.cat([self.ense_coe[i] * hbi for i, hbi in enumerate(h_bands)], dim=1)
            else:
                h = torch.cat(h_bands, dim=1)
        elif self.combine == 'lp':
            h = h_bands[0]
        elif self.combine == 'hp':
            h = h_bands[-1]
        else:
            # default to sum
            h = sum(h_bands)

        # MLP stack
        for i in range(self.nl):
            h = self.layers[i](h)
            if i < self.nl - 1:
                if self.acti:
                    h = F.relu(h)
                h = F.dropout(h, self.drop[1], self.training)

        return h

class TFE_GNN_large(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout, activation, hop, combine):
        super(TFE_GNN_large, self).__init__()
        self.nl = num_layers
        self.drop = dropout
        self.acti = activation
        self.hops = hop if isinstance(hop, (list, tuple)) else [hop]
        self.combine = combine

        self.adaptives = nn.ParameterList([nn.Parameter(torch.Tensor(h + 1)) for h in self.hops])
        self.adaptive_lp = self.adaptives[0]
        self.adaptive = self.adaptives[-1]

        if self.combine in ('sum', 'con'):
            self.ense_coe = nn.Parameter(torch.Tensor(len(self.adaptives)))

        self.nlayers_input_dim = input_dim * (len(self.adaptives) if self.combine == 'con' else 1)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(self.nlayers_input_dim, hidden_dim, bias=False))
            elif i < num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            else:
                self.layers.append(nn.Linear(hidden_dim, out_dim, bias=False))

        self.init_parameter()

    def init_parameter(self):
        if hasattr(self, 'ense_coe'):
            self.ense_coe.data.fill_(1.0)
        for p in self.adaptives:
            p.data.fill_(0.5)
        for i in range(self.nl):
            stdv = 1. / math.sqrt(self.layers[i].weight.size(1))
            self.layers[i].weight.data.normal_(-stdv, stdv)

    def mix_prop(self, adj, x, coe):
        x0 = x.clone()
        xx = x.clone() * coe[0]
        for i in range(1, len(coe)):
            x0 = torch.mm(adj, x0)
            xx = xx + coe[i] * x0
        return xx

    def forward(self, adj_hp, adj_lp, h1):
        if isinstance(adj_hp, (list, tuple)):
            adjs = list(adj_hp)
            adaptives = list(self.adaptives)
            bands = min(len(adaptives), len(adjs))
            adaptives = adaptives[:bands]
            adjs = adjs[:bands]
            h0 = h1
        else:
            adjs = [adj_lp, adj_hp]
            adaptives = [self.adaptive_lp, self.adaptive]
            h0 = h1

        if self.drop[0] > 0:
            h0 = F.dropout(h0, self.drop[0], self.training)

        h_bands = [self.mix_prop(A, h0, coe) for A, coe in zip(adjs, adaptives)]

        if self.combine == 'sum':
            if hasattr(self, 'ense_coe'):
                h = sum(self.ense_coe[i] * hbi for i, hbi in enumerate(h_bands))
            else:
                h = sum(h_bands)
        elif self.combine == 'con':
            if hasattr(self, 'ense_coe'):
                h = torch.cat([self.ense_coe[i] * hbi for i, hbi in enumerate(h_bands)], dim=1)
            else:
                h = torch.cat(h_bands, dim=1)
        elif self.combine == 'lp':
            h = h_bands[0]
        elif self.combine == 'hp':
            h = h_bands[-1]
        else:
            h = sum(h_bands)

        return h