import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GATLayer(Module):
    def __init__(self, input_channel, output_channel, use_bias=True):
        super(GATLayer, self).__init__()
        self.use_bias = use_bias
        self.input_channel = input_channel
        self.output_channel = output_channel

        ### Layer kernel
        self.kernel = Parameter(torch.FloatTensor(input_channel, output_channel))
        torch.nn.init.xavier_uniform_(self.kernel, gain=1.414)

        ### Layer bias
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(output_channel))
            torch.nn.init.constant_(self.bias, 0)

        ### Attention kernels
        self.attn_kernel_self = Parameter(torch.FloatTensor(output_channel, 1))
        torch.nn.init.xavier_uniform_(self.attn_kernel_self, gain=1.414)
        self.attn_kernel_neighs = Parameter(torch.FloatTensor(output_channel, 1))
        torch.nn.init.xavier_uniform_(self.attn_kernel_neighs, gain=1.414)

    def forward(self, X, adj):
        # feature embedding
        # (N x F')
        features = torch.mm(X, self.kernel)
        # (N x 1)
        attn_for_self = torch.mm(features, self.attn_kernel_self)
        attn_for_neigh = torch.mm(features, self.attn_kernel_neighs)

        # attention head: a(Wh_i, Wh_j)
        attn = attn_for_self + attn_for_neigh.t()
        attn = F.leaky_relu(attn, negative_slope=0.2)

        # mask values before activation
        mask = -10e9 * torch.ones_like(attn)
        attn = torch.where(adj>0, attn, mask)

        # graph attention weights
        # attn:(N * N)
        attn = torch.softmax(attn, dim=-1)

        # Apply dropout to features and attention coefficients
        attn = F.dropout(attn, p=0.5, training=self.training)
        features = F.dropout(features, p=0.5, training=self.training)

        # Linear combination with neighbors' features
        node_features = torch.mm(attn, features)

        if self.use_bias:
            node_features = node_features + self.bias

        # if self.attn_heads_reduction == 'concat':
        #     output = torch.cat(out)
        # else:
        #     output = torch.mean(torch.stack(out), dim=0)
        # output = F.relu(output)
        return node_features


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_head):
        super(GAT, self).__init__()
        self.attn_heads_reduction = 'concat'
        self.gc1 = [GATLayer(nfeat, nhid) for _ in range(num_head[0])]
        if self.attn_heads_reduction == 'concat':
            self.gc2 = [GATLayer(nhid*num_head[0], nclass) for _ in range(num_head[1])]
        else:
            self.gc2 = [GATLayer(nhid, nclass) for _ in range(num_head[1])]
        self.gc1 = nn.ModuleList(self.gc1)
        self.gc2 = nn.ModuleList(self.gc2)
        self.dropout = dropout

    def forward(self, x, adj):
        out = []
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(len(self.gc1)):
            res = F.relu(self.gc1[i](x, adj))
            out.append(res)
        if self.attn_heads_reduction == 'concat':
            out = torch.cat(out, dim=1)
        else:
            out = torch.mean(torch.stack(out), dim=0)
        out = F.elu(out)

        x = F.dropout(out, self.dropout, training=self.training)
        out = []
        for i in range(len(self.gc2)):
            res = F.relu(self.gc2[i](x, adj))
            out.append(res)
        if self.attn_heads_reduction == 'concat':
            out = torch.cat(out)
        else:
            out = torch.mean(torch.stack(out), dim=0)
        out = F.relu(out)

        return F.log_softmax(out, dim=1)
