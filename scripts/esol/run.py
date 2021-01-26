import numpy as np
import torch
import dgl
import stag

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

        # local import
        from dgl.nn import pytorch as dgl_nn

        # get activation function from pytorch if specified
        if activation is not None:
            activation = getattr(torch.nn.functional, activation)

        # initial layer: in -> hidden
        self.gn0 = layer(
            in_feats=in_features,
            out_feats=hidden_features,
            activation=activation,
            allow_zero_in_degree=True,
        )

        # last layer: hidden -> hidden
        setattr(
            self,
            "gn%s" % (depth-1),
            layer(
                in_feats=hidden_features,
                out_feats=hidden_features,
                activation=activation,
                allow_zero_in_degree=True,
            )
        )

        # middle layers: hidden -> hidden
        for idx in range(1, depth-1):
            setattr(
                self,
                "gn%s" % idx,
                layer(
                    in_feats=hidden_features,
                    out_feats=hidden_features,
                    activation=activation,
                    allow_zero_in_degree=True,
                )
            )


        self.d = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features),
        )

        self.depth = depth

    def forward(self, g, x):
        # ensure local scope
        g = g.local_var()

        for idx in range(self.depth):
            x = getattr(self, "gn%s" % idx)(g, x)

        g.ndata["x"] = x
        x = dgl.readout.sum_nodes(g, "x")
        x = self.d(x)
        return x

def rmse(y_pred, y_true):
    return torch.nn.MSELoss()(y_true, y_pred).pow(0.5)

def run(args):
    import dgl.nn.pytorch as dgl_nn
    from functools import partial
    from dgllife.data import ESOL
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    if args.stag != "none":
        dgl.function.copy_src = dgl.function.copy_u = partial(
            getattr(
                stag,
                "stag_copy_src_%s" % args.stag
            ),
            alpha=args.alpha
        )
    
    
    if args.layer == "SAGEConv":
        layer = partial(dgl_nn.SAGEConv, aggregator_type="mean")

    elif args.layer == "GINConv":
        class Layer(torch.nn.Module):
            def __init__(self, in_feats, out_feats, activation):
                super(Layer, self).__init__()
                if activation == None:
                    activation = lambda x: x
                self.d0 = torch.nn.Linear(in_feats, in_feats)
                self.activation = activation
                self.d1 = torch.nn.Linear(in_feats, out_feats)

            def forward(self, x):
                x = self.d0(x)
                x = self.activation(x)
                x = self.d1(x)
                x = self.activation(x)
                return x

        layer = lambda in_feats, out_feats, allow_zero_in_degree, activation: dgl_nn.GINConv(
            apply_func=Layer(in_feats, out_feats, activation=activation),
            aggregator_type="sum",
        )

    else:
        layer = getattr(dgl_nn, args.layer)


    ds = ESOL(smiles_to_bigraph, CanonicalAtomFeaturizer())
    ds = list(ds)
    import random
    random.Random(2666).shuffle(ds)
    _, g, y = zip(*ds)
    n_data = len(g)

    g_tr, y_tr = g[:int(0.8*n_data)], y[:int(0.8*n_data)]
    g_te, y_te = g[int(0.8*n_data):int(0.9*n_data)], y[int(0.8*n_data):int(0.9*n_data)]
    g_vl, y_vl = g[int(0.9*n_data):], y[int(0.9*n_data):]

    g_tr = dgl.batch(g_tr)
    g_te = dgl.batch(g_te)
    g_vl = dgl.batch(g_vl)

    y_tr = torch.stack(y_tr)
    y_te = torch.stack(y_te)
    y_vl = torch.stack(y_vl)


    net = Net(
        layer=layer,
        in_features=74,
        out_features=1,
        hidden_features=args.hidden_features,
        activation=args.activation,
        depth=args.depth,
    )

    optimizer = torch.optim.Adam(net.parameters(), args.lr)

    if torch.cuda.is_available():
        net = net.to('cuda:0')
        g_tr = g_tr.to('cuda:0')
        g_te = g_te.to('cuda:0')
        g_vl = g_vl.to('cuda:0')
        y_tr = y_tr.to('cuda:0')
        y_te = y_te.to('cuda:0')
        y_vl = y_vl.to('cuda:0')

    def sampling_performance(net, n_samples=16):
        _accuracy_tr = rmse(
            torch.mean(torch.stack([
                net(g_tr, g_tr.ndata['h'])
                for _ in range(n_samples)
            ], dim=0), dim=0),
            y_tr,
        ).item()

        _accuracy_te = rmse(
            torch.mean(torch.stack([
                net(g_te, g_te.ndata['h'])
                for _ in range(n_samples)
            ], dim=0), dim=0),
            y_te,
        ).item()

        _accuracy_vl = rmse(
            torch.mean(torch.stack([
                net(g_vl, g_vl.ndata['h'])
                for _ in range(n_samples)
            ], dim=0), dim=0),
            y_vl,
        ).item()

        return _accuracy_tr, _accuracy_te, _accuracy_vl


    accuracy_tr = []
    accuracy_te = []
    accuracy_vl = []

    for idx_epoch in range(args.n_epochs):
        optimizer.zero_grad()
        y_pred = net(g_tr, g_tr.ndata['h'])
        y_true = y_tr
        loss = torch.nn.MSELoss()(y_pred, y_true)
        loss.backward()
        optimizer.step()

        if idx_epoch % args.report_interval == 0:
            _accuracy_tr, _accuracy_te, _accuracy_vl = sampling_performance(net, n_samples=16)

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


    _accuracy_tr, _accuracy_te, _accuracy_vl = sampling_performance(net, n_samples=16)


    import pandas as pd
    df = pd.DataFrame.from_dict(
        {
            "accuracy_tr": [accuracy_tr[-1]],
            "accuracy_te": [accuracy_te[-1]],
            "accuracy_vl": [accuracy_vl[-1]],
            "sampled_tr": _accuracy_tr,
            "sampled_te": _accuracy_te,
            "sampled_vl": _accuracy_vl,
        }
    )

    df.to_markdown(open(args.out + "/overview.md", "w"))

    # torch.save(net.to('cpu'), args.out + "/net.th")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--stag", type=str, default="none")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--layer", type=str, default="GraphConv")
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--report_interval", type=int, default=100)
    args = parser.parse_args()
    run(args)
