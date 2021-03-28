import numpy as np
import torch
import dgl
import stag

import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=False,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=True,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))


            graph.edata["a"] = torch.distributions.Normal(
                torch.tensor(1.0, device=graph.edata["a"].device),
                torch.tensor(args.alpha, device=graph.edata["a"].device),
            ).rsample(graph.edata["a"].shape) * graph.edata["a"]

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)

            return rst

import numpy as np
import torch
import dgl

class ElementWiseLinear(torch.nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = torch.nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class Net(torch.nn.Module):
    def __init__(
            self,
            layer,
            in_features,
            hidden_features,
            out_features,
            depth,
            activation=None
        ):
        super(Net, self).__init__()

        # get activation function from pytorch if specified
        if activation is not None:
            activation = getattr(torch.nn.functional, activation)

        self.activation = activation

        _hidden_features = int(hidden_features / 1)


        # initial layer: in -> hidden
        self.gn0 = layer(
            in_feats=in_features,
            out_feats=_hidden_features,
        )

        self.bn0 = torch.nn.BatchNorm1d(hidden_features)

        # last layer: hidden -> out
        setattr(
            self,
            "gn%s" % (depth-1),
            layer(
                in_feats=hidden_features,
                out_feats=out_features,
            )
        )

        # middle layers: hidden -> hidden
        for idx in range(1, depth-1):
            setattr(
                self,
                "gn%s" % idx,
                layer(
                    in_feats=hidden_features,
                    out_feats=_hidden_features,
                )
            )

            setattr(
                self,
                "bn%s" % idx,
                torch.nn.BatchNorm1d(hidden_features),
            )

        self.bias_last = ElementWiseLinear(out_features, weight=False, bias=True, inplace=True)

        self.depth = depth

    def forward(self, g, x):
        # ensure local scope
        g = g.local_var()
        x = self.gn0(g, x)
        x = x.flatten(1)
        x = self.bn0(x)
        x = self.activation(x)


        for idx in range(1, self.depth-1):
            x = getattr(self, "gn%s" % idx)(g, x)
            x = x.flatten(1)
            x = getattr(self, "bn%s" % idx)(x)
            x = self.activation(x)

        if self.depth > 1:
            x = getattr(self, "gn%s" % (self.depth-1))(g, x)
            x = x.mean(dim=1)

        x = self.bias_last(x)

        return x



def accuracy_between(y_pred, y_true):
    if y_pred.dim() >= 2:
        y_pred = y_pred.argmax(dim=-1)
    return (y_pred == y_true).sum() / y_pred.shape[0]

def run(args):
    from functools import partial
    from dgl.nn import pytorch as dgl_nn

    if args.stag.startswith("none") is False:
        dgl.function.copy_src = dgl.function.copy_u = partial(
            getattr(
                stag,
                "stag_copy_src_%s" % args.stag
            ),
            alpha=args.alpha
        )

    elif args.stag == "none_de":
        dgl.function.sum = partial(
            getattr(
                stag,
                "stag_sum_bernoulli_shared"
            ),
            alpha=args.alpha
        )

    elif args.stag == "none_gdc":
        dgl.function.sum = partial(
            getattr(
                stag,
                "stag_sum_bernoulli"
            ),
            alpha=args.alpha
        )

    elif args.stag == "none_dropout":
        dgl.function.sum = partial(
            stag.stag_sum_dropout,
            alpha=args.alpha
        )
    from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
    evaluator = Evaluator(name='ogbn-arxiv')
    dataset = DglNodePropPredDataset(name=args.data)
    split_idx = dataset.get_idx_split()
    g, labels = dataset[0]
    feats = g.ndata['feat']# .to("cuda:0")
    labels = labels# .to("cuda:0")
    train_idx = split_idx['train']# .to("cuda:0")
    test_idx = split_idx['test']# .to("cuda:0")
    valid_idx = split_idx['valid']# .to("cuda:0")

    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)
    g = g.int()# .to("cuda:0")

    def sampling_performance(g, net, n_samples=16):

        y_pred = torch.stack([net(g, feats).detach().softmax(dim=-1) for _ in range(n_samples)], dim=0).mean(dim=0).argmax(dim=-1, keepdim=True)
        y_true = labels

        _accuracy_tr = evaluator.eval({"y_true": y_true[train_idx], "y_pred": y_pred[train_idx]})["acc"]
        _accuracy_te = evaluator.eval({"y_true": y_true[test_idx], "y_pred": y_pred[test_idx]})["acc"]
        _accuracy_vl = evaluator.eval({"y_true": y_true[valid_idx], "y_pred": y_pred[valid_idx]})["acc"]

        return _accuracy_tr, _accuracy_te, _accuracy_vl

    import functools
    layer = functools.partial(
            dgl.nn.GATConv,
            num_heads=1
    )

    net = Net(
        layer=layer,
        in_features=feats.size(-1),
        out_features=dataset.num_classes,
        hidden_features=args.hidden_features,
        activation=args.activation,
        depth=args.depth,
    )# .to("cuda:0")

    if torch.cuda.is_available():
        net = net.cuda()

    import itertools
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    accuracy_tr = []
    accuracy_te = []
    accuracy_vl = []

    for idx_epoch in range(args.n_epochs):
        net.train()
        optimizer.zero_grad()
        out = net(g, feats)[train_idx].log_softmax(dim=-1)
        loss = torch.nn.functional.nll_loss(out, labels.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        net.eval()

        if idx_epoch % args.report_interval == 0:
            _accuracy_tr, _accuracy_te, _accuracy_vl = sampling_performance(g, net, n_samples=32)

        accuracy_tr.append(_accuracy_tr)
        accuracy_te.append(_accuracy_te)
        accuracy_vl.append(_accuracy_vl)

    accuracy_tr = np.array(accuracy_tr)
    accuracy_te = np.array(accuracy_te)
    accuracy_vl = np.array(accuracy_vl)

    import os
    os.mkdir(args.out)
    np.save(args.out + "/accuracy_tr.npy", accuracy_tr)
    np.save(args.out + "/accuracy_vl.npy", accuracy_vl)
    np.save(args.out + "/accuracy_te.npy", accuracy_te)

    best_epoch = accuracy_vl.argmax()
    import pandas as pd
    df = pd.DataFrame.from_dict(
        {
            "tr": [accuracy_tr[best_epoch]],
            "te": [accuracy_te[best_epoch]],
            "vl": [accuracy_vl[best_epoch]],
        },
    )

    df.to_markdown(open(args.out + "/overview.md", "w"))

    torch.save(net.to('cpu'), args.out + "/net.th")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--stag", type=str, default="none")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--layer", type=str, default="GraphConv")
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--report_interval", type=int, default=100)
    parser.add_argument("--data", type=str, default="ogbn-arxiv")
    args = parser.parse_args()
    run(args)
