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
        self.f_z_mu = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_features, 1),
        )
        self.f_z_log_sigma = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_features, 1),
        )

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
                out_feats=out_features,
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

        self.gn0_enc = layer(in_feats=in_features, out_feats=hidden_features, activation=activation)
        self.gn1_enc = layer(in_feats=hidden_features, out_feats=hidden_features, activation=activation)

        if depth == 1:
            self.gn0 = layer(
                in_feats=in_features, out_feats=out_features, activation=None
            )

        self.depth = depth
        
        assert g_ref is not None
        self.a_prior = torch.distributions.Normal(1.0, args.a_prior)
        self.in_features = in_features
        self.hidden_features = hidden_features

    def get_reconstruction_loss(self, g, x):
        g = g.local_var()
        z_node = self.gn0_enc(g, x)
        z_node = self.gn1_enc(g, z_node)

        positive_idxs = g.adjacency_matrix().coalesce().indices()
        negative_idxs = torch.cat(
            [
                torch.randint(low=0, high=g.number_of_nodes()-1, size=(1, g.number_of_edges())),
                torch.randint(low=0, high=g.number_of_nodes()-1, size=(1, g.number_of_edges())),
            ],
            dim=0,
        )

        z_edge_positive = torch.cat([z_node[positive_idxs[0]], z_node[positive_idxs[1]]], dim=-1)
        z_edge_negative = torch.cat([z_node[negative_idxs[0]], z_node[negative_idxs[1]]], dim=-1)

        posterior_positive = torch.distributions.Normal(
            self.f_z_mu(z_edge_positive).squeeze(),
            self.f_z_log_sigma(z_edge_positive).squeeze().exp()
        )

        posterior_negative = torch.distributions.Normal(
            self.f_z_mu(z_edge_negative).squeeze(),
            self.f_z_log_sigma(z_edge_negative).squeeze().exp(),
        )

        return -posterior_positive.log_prob(torch.ones(g.number_of_edges(), device=x.device)).mean() - posterior_negative.log_prob(torch.zeros(g.number_of_edges(), device=x.device)).mean()

    def forward(self, g, x):
        # ensure local scope
        g = g.local_var()
        
        z_node = self.gn0_enc(g, x)
        z_node = self.gn1_enc(g, z_node)
        g.ndata["z_node"] = z_node

        g.apply_edges(
            lambda edges: {
                "a_first": torch.distributions.Normal(
                    self.f_z_mu(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).squeeze(),
                    self.f_z_log_sigma(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).exp().squeeze()
                ).rsample([self.in_features]).transpose(1, 0),
                "a_rest": torch.distributions.Normal(
                    self.f_z_mu(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).squeeze(),
                    self.f_z_log_sigma(torch.cat([edges.src["z_node"], edges.dst["z_node"]], dim=-1)).exp().squeeze()
                ).rsample([self.hidden_features]).transpose(1, 0)
            }
        )

        # nll_reg = -self.a_prior.log_prob(g.edata["a_first"]).mean() - self.a_prior.log_prob(g.edata["a_rest"]).mean()
        nll_reg = self.get_reconstruction_loss(g, x)

        g.edata["a"] = g.edata["a_first"]
        x = self.gn0(g, x)

        for idx in range(1, self.depth):
            g.edata["a"] = g.edata["a_rest"]
            x = getattr(self, "gn%s" % idx)(g, x)

        return x, nll_reg

def accuracy_between(y_pred, y_true):
    if y_pred.dim() >= 2:
        y_pred = y_pred.argmax(dim=-1)
    return (y_pred == y_true).sum() / y_pred.shape[0]

def sampling_performance(g, net, n_samples=16):
    y_pred = torch.stack([net(g, g.ndata['feat'])[0].detach() for _ in range(n_samples)], dim=0).mean(dim=0)
    y_true = g.ndata['label']

    _accuracy_tr = accuracy_between(y_pred[g.ndata['train_mask']], y_true[g.ndata['train_mask']]).item()
    _accuracy_te = accuracy_between(y_pred[g.ndata['test_mask']], y_true[g.ndata['test_mask']]).item()
    _accuracy_vl = accuracy_between(y_pred[g.ndata['val_mask']], y_true[g.ndata['val_mask']]).item()

    return _accuracy_tr, _accuracy_te, _accuracy_vl

def run(args):
    from functools import partial
    from dgl.nn import pytorch as dgl_nn
    
    def stag_copy_src_vi(src="h", out="m"):
        def message_fun(edges):
            if "a" in edges.data:
                return {out: edges.src[src] * edges.data["a"]}
            return {out: edges.src[src]}

        return message_fun

    dgl.function.copy_src = dgl.function.copy_u = stag_copy_src_vi 
    
    ds = dgl.data.CoraGraphDataset()
    g = ds[0]
    layer = getattr(dgl_nn, args.layer)
    net = Net(
        layer=layer,
        in_features=1433,
        out_features=7,
        hidden_features=args.hidden_features,
        activation=args.activation,
        depth=args.depth,
        g_ref = g,
    )

    
    import itertools
    optimizer = torch.optim.Adam(
        [
            {"params": net.gn0.parameters(), "lr": args.lr, "weight_decay": 5e-3},
            {
                "params": itertools.chain(
                    *[getattr(net, "gn%s" % idx).parameters() for idx in range(1, args.depth)]
                ),
                "lr": args.lr
            },
            {
                "params": itertools.chain(
                    net.gn0_enc.parameters(),
                    net.gn1_enc.parameters(),
                    net.f_z_mu.parameters(),
                    net.f_z_log_sigma.parameters(),
                ),
                "lr": args.lr_vi
            }
        ]

    )

    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    if torch.cuda.is_available():
        net = net.to('cuda:0')
        g = g.to('cuda:0')

    accuracy_tr = []
    accuracy_te = []
    accuracy_vl = []
    
    for idx_epoch in range(args.n_epochs):
        net.train()
        optimizer.zero_grad()
        y_pred, nll_reg = net(g, g.ndata['feat'])
        y_pred = y_pred[g.ndata['train_mask']]
        y_true = g.ndata['label'][g.ndata['train_mask']]

        loss = torch.nn.functional.nll_loss(y_pred.log_softmax(dim=-1), y_true) + nll_reg
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
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--stag", type=str, default="none")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--layer", type=str, default="GraphConv")
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--report_interval", type=int, default=100)
    parser.add_argument("--a_prior", type=float, default=1.0)
    parser.add_argument("--a_log_sigma_init", type=float, default=-1.0)
    parser.add_argument("--a_mu_init_std", type=float, default=1.0)
    parser.add_argument("--lr_vi", type=float, default=1e-3)
    args = parser.parse_args()
    run(args)
