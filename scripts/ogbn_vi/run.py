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

        if depth == 1:
            self.gn0 = layer(
                in_feats=in_features, out_feats=out_features, activation=None
            )

        self.depth = depth

        self.a_prior = torch.distributions.Normal(1.0, args.a_prior)

        # middle layers
        self.a_mu = torch.nn.Parameter(
            torch.distributions.Normal(1, args.a_mu_init_std).sample(
                (depth-1, hidden_features)
            )
        )

        self.a_log_sigma = torch.nn.Parameter(
            args.a_log_sigma_init * torch.ones(depth-1, hidden_features)
        )

        # first layer
        self.a_mu_first = torch.nn.Parameter(
            torch.distributions.Normal(1, args.a_mu_init_std).sample(
                [in_features]
            )
        )

        self.a_log_sigma_first = torch.nn.Parameter(
            args.a_log_sigma_init * torch.ones(in_features)
        )

        self.in_features = in_features


    def forward(self, g, x):
        # ensure local scope
        g = g.local_var()

        # z_node = self.gn0_enc(g, x)
        # z_node = self.gn1_enc(g, z_node)
        # g.ndata["z_node"] = z_node

        g.apply_edges(
            lambda edges: {
                "a_first": torch.distributions.Normal(
                    self.a_mu_first, self.a_log_sigma_first,
                    ).rsample([g.number_of_edges()]),
                "a_rest": torch.distributions.Normal(
                    self.a_mu, self.a_log_sigma,
                    ).rsample([g.number_of_edges()]),
            }
        )

        nll_reg = -self.a_prior.log_prob(g.edata["a_first"]).mean() - self.a_prior.log_prob(g.edata["a_rest"]).mean()

        g.edata["a"] = g.edata["a_first"]
        x = self.gn0(g, x)

        for idx in range(1, self.depth):
            g.edata["a"] = g.edata["a_rest"][:, idx-1, :]
            x = getattr(self, "gn%s" % idx)(g, x)


        return x, nll_reg



def accuracy_between(y_pred, y_true):
    if y_pred.dim() >= 2:
        y_pred = y_pred.argmax(dim=-1)
    return (y_pred == y_true).sum() / y_pred.shape[0]

def run(args):
    from functools import partial
    from dgl.nn import pytorch as dgl_nn

    dgl.function.copy_src = dgl.function.copy_u = stag.stag_copy_src_vi

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
    g = g.int()#.to("cuda:0")
 
    def sampling_performance(g, net, n_samples=16):

        y_pred = torch.stack([net(g, feats)[0].detach() for _ in range(n_samples)], dim=0).mean(dim=0).argmax(dim=-1, keepdim=True)
        y_true = labels
        
        _accuracy_tr = evaluator.eval({"y_true": y_true[train_idx], "y_pred": y_pred[train_idx]})["acc"]
        _accuracy_te = evaluator.eval({"y_true": y_true[test_idx], "y_pred": y_pred[test_idx]})["acc"]
        _accuracy_vl = evaluator.eval({"y_true": y_true[valid_idx], "y_pred": y_pred[valid_idx]})["acc"]

        return _accuracy_tr, _accuracy_te, _accuracy_vl

    layer = dgl.nn.GraphConv

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
        g = g.to("cuda:0")
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()
        valid_idx = valid_idx.cuda()
        feats = feats.cuda()
        labels = labels.cuda()



    import itertools
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    accuracy_tr = []
    accuracy_te = []
    accuracy_vl = []

    net.train()

    for idx_epoch in range(args.n_epochs):
        optimizer.zero_grad()
        out, nll_reg = net(g, feats)
        out = out[train_idx].log_softmax(dim=-1)
        loss = torch.nn.functional.nll_loss(out, labels.squeeze(1)[train_idx]) + nll_reg
        loss.backward()
        optimizer.step()
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

    parser.add_argument("--a_prior", type=float, default=1.0)
    parser.add_argument("--a_log_sigma_init", type=float, default=-1.0)
    parser.add_argument("--a_mu_init_std", type=float, default=1.0)
    parser.add_argument("--lr_vi", type=float, default=1e-3)

    args = parser.parse_args()
    run(args)
