import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class GatedGCN(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout=0.0, batch_norm=True, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)


    def forward(self, g, h, edge_weight=None):

        h_in = h # for residual connection

        aggregate_fn = fn.copy_src('h', 'm')
        if edge_weight is not None:
            assert edge_weight.shape[0] == g.number_of_edges()
            g.edata['_edge_weight'] = edge_weight
            aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
        else:
            aggregate_fn = fn.copy_u('Bh', 'm')

        g.ndata['h']  = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)

        g.update_all(aggregate_fn, fn.sum('m', 'sum_h'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_h']
        h = g.ndata['h'] # result of graph convolution

        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization

        h = F.relu(h) # non-linear activation

        if self.residual:
            h = h_in + h # residual connection

        h = F.dropout(h, self.dropout)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
