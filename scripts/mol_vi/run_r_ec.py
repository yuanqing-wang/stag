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
            activation=None,
            g_ref=None,
        ):
        super(Net, self).__init__()

        # get activation function from pytorch if specified
        if activation is not None:
            activation = getattr(torch.nn.functional, activation)

        self.gn0_enc = layer(in_feats=in_features, out_feats=hidden_features, activation=activation)
        self.gn1_enc = layer(in_feats=hidden_features, out_feats=hidden_features, activation=activation)

        self.f_z_mu = torch.nn.Linear(2*hidden_features, hidden_features)
        self.f_z_log_sigma = torch.nn.Linear(2*hidden_features, hidden_features)
        self.f_z_mu_first = torch.nn.Linear(2*hidden_features, in_features)
        self.f_z_log_sigma_first = torch.nn.Linear(2*hidden_features, in_features)

        torch.nn.init.normal_(self.f_z_log_sigma.weight, std=1e-3)
        torch.nn.init.constant_(self.f_z_log_sigma.bias, args.a_log_sigma_init)
        torch.nn.init.normal_(self.f_z_log_sigma_first.weight, std=1e-3)
        torch.nn.init.constant(self.f_z_log_sigma_first.bias, args.a_log_sigma_init)

        torch.nn.init.normal_(self.f_z_mu.weight, std=1e-2)
        torch.nn.init.normal_(self.f_z_mu.bias, mean=1.0, std=args.a_mu_init_std)
        torch.nn.init.normal_(self.f_z_mu_first.weight, std=1e-2)
        torch.nn.init.normal_(self.f_z_mu_first.bias, mean=1.0, std=args.a_mu_init_std)


        # initial layer: in -> hidden
        self.gn0 = layer(
            in_feats=in_features,
            out_feats=hidden_features,
            activation=activation,
        )

        # last layer: hidden -> out
        setattr(
            self,
            "gn%s" % (depth-1),
            layer(
                in_feats=hidden_features,
                out_feats=hidden_features,
                activation=None,
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
                    activation=activation
                )
            )

        if depth == 1:
            self.gn0 = layer(
                in_feats=in_features, out_feats=out_features, activation=None
            )

        self.depth = depth

        self.d = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features),
        )

        self.a_prior = torch.distributions.Normal(1.0, args.a_prior)

    def forward(self, g, x):
        # ensure local scope
        g = g.local_var()

        z_node = self.gn0_enc(g, x)
        z_node = self.gn1_enc(g, z_node)
        g.ndata["z_node"] = z_node

        g.apply_edges(
            lambda edges: {
                "a_first": torch.distributions.Normal(
                    self.f_z_mu_first(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)),
                    self.f_z_log_sigma_first(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).exp()
                ).rsample(),
                "a_rest": torch.distributions.Normal(
                    self.f_z_mu(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)),
                    self.f_z_log_sigma(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).exp()
                ).rsample()
            }
        )

        nll_reg = -self.a_prior.log_prob(g.edata["a_first"]).mean() - self.a_prior.log_prob(g.edata["a_rest"]).mean()

        g.edata["a"] = g.edata["a_first"]
        x = self.gn0(g, x)

        for idx in range(1, self.depth):
            g.edata["a"] = g.edata["a_rest"]
            x = getattr(self, "gn%s" % idx)(g, x)

        g.ndata['x'] = x
        x = dgl.sum_nodes(g, "x")
        x = self.d(x)

        return x, nll_reg

def rmse(y_pred, y_true):
    return torch.nn.MSELoss()(y_true, y_pred).pow(0.5)

def run(args):
    import dgl.nn.pytorch as dgl_nn
    from functools import partial
    from dgllife.data import ESOL, Lipophilicity, FreeSolv
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    dgl.function.copy_src = dgl.function.copy_u = stag.stag_copy_src_vi

    if args.data == "ESOL":
        ds = ESOL(smiles_to_bigraph, CanonicalAtomFeaturizer())
    elif args.data == "Lipophilicity":
        ds = Lipophilicity(smiles_to_bigraph, CanonicalAtomFeaturizer())
    elif args.data == "FreeSolv":
        ds = FreeSolv(smiles_to_bigraph, CanonicalAtomFeaturizer())

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

    import functools
    layer = functools.partial(
            getattr(dgl_nn, args.layer),
            allow_zero_in_degree=True
    )

    net = Net(
        layer=layer,
        in_features=74,
        out_features=1,
        hidden_features=args.hidden_features,
        activation=args.activation,
        depth=args.depth,
    )

    import itertools
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

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
                net(g_tr, g_tr.ndata['h'])[0].detach()
                for _ in range(n_samples)
            ], dim=0), dim=0),
            y_tr,
        ).item()

        _accuracy_te = rmse(
            torch.mean(torch.stack([
                net(g_te, g_te.ndata['h'])[0].detach()
                for _ in range(n_samples)
            ], dim=0), dim=0),
            y_te,
        ).item()

        _accuracy_vl = rmse(
            torch.mean(torch.stack([
                net(g_vl, g_vl.ndata['h'])[0].detach()
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
        y_pred, nll_reg = net(g_tr, g_tr.ndata['h'])
        y_true = y_tr
        loss = torch.nn.MSELoss()(y_pred, y_true) + nll_reg
        loss.backward()
        optimizer.step()

        if idx_epoch % args.report_interval == 0:
            _accuracy_tr, _accuracy_te, _accuracy_vl = sampling_performance(net, n_samples=32)

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

    best_epoch = accuracy_vl.argmin()

    import pandas as pd
    df = pd.DataFrame.from_dict(
        {
            "accuracy_tr": [accuracy_tr[best_epoch]],
            "accuracy_te": [accuracy_te[best_epoch]],
            "accuracy_vl": [accuracy_vl[best_epoch]],
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
    parser.add_argument("--data", type=str, default="ESOL")
    parser.add_argument("--a_log_sigma_init", type=float, default=1.0)
    parser.add_argument("--a_mu_init_std", type=float, default=1.0)
    parser.add_argument("--a_prior", type=float, default=1.0)
    args = parser.parse_args()
    run(args)
