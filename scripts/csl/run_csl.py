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
        self.gn0 = getattr(dgl_nn, layer)(
            in_feats=in_features,
            out_feats=hidden_features,
            activation=activation,
        )

        # last layer: hidden -> out
        setattr(
            self,
            "gn%s" % (depth-1),
            getattr(dgl_nn ,layer)(
                in_feats=hidden_features,
                out_feats=out_features,
            )
        )

        # middle layers: hidden -> hidden
        for idx in range(1, depth-1):
            setattr(
                self,
                "gn%s" % idx,
                getattr(dgl_nn, layer)(
                    in_feats=hidden_features,
                    out_feats=hidden_features,
                    activation=activation
                )
            )

        self.depth = depth

    def forward(self, g, x):
        # ensure local scope
        g = g.local_var()

        for idx in range(self.depth):
            x = getattr(self, "gn%s" % idx)(g, x)

        g.ndata["x"] = x
        x = dgl.readout.sum_nodes(g, "x")
        return x

def accuracy_between(y_pred, y_true):
    if y_pred.dim() >= 2:
        y_pred = y_pred.argmax(dim=-1)
    return (y_pred == y_true).sum() / y_pred.shape[0]


def run(args):

    import urllib
    import os
    if not os.exist("graphs_Kary_Deterministic_Graphs.pkl"):
        urllib.urlretrieve(
            "https://github.com/PurdueMINDS/RelationalPooling/raw/master/Synthetic_Data/graphs_Kary_Deterministic_Graphs.pkl",
            "graphs_Kary_Deterministic_Graphs.pkl"
        )

        urllib.urlretrieve(
            "https://github.com/PurdueMINDS/RelationalPooling/raw/master/Synthetic_Data/y_Kary_Deterministic_Graphs.pt",
            "y_Kary_Deterministic_Graphs.pt"
        )

    import pickle
    adjs = pickle.load(open("/content/graphs_Kary_Deterministic_Graphs.pkl", "rb"))
    labels = torch.load("/content/y_Kary_Deterministic_Graphs.pt")
    graphs = []
    for adj in adjs:
        g = dgl.DGLGraph()
        g = dgl.from_scipy(adj)
        graphs.append(g)

    f_handle = open("idxs.txt", "r")
    tr_line, te_line, vl_line = f_handle.readlines()
    idx_tr = [int(x) for x in tr_line.split(",")]
    idx_te = [int(x) for x in te_line.split(",")]
    idx_vl = [int(x) for x in vl_line.split(",")]

    g_tr = dgl.batch([graphs[idx] for idx in idx_tr])
    g_te = dgl.batch([graphs[idx] for idx in idx_te])
    g_vl = dgl.batch([graphs[idx] for idx in idx_vl])

    y_tr = labels[idx_tr]
    y_te = labels[idx_te]
    y_vl = labels[idx_vl]

    def sampling_performance(net, n_samples=16):
        _accuracy_tr = accuracy_between(
            net(g_tr, torch.zeros(g_tr.number_of_nodes(), 1, device=g.device)),
            y_tr,
        ).item()

        _accuracy_te = accuracy_between(
            net(g_te, torch.zeros(g_te.number_of_nodes(), 1, device=g.device)),
            y_te,
        ).item()

        _accuracy_vl = accuracy_between(
            net(g_vl, torch.zeros(g_vl.number_of_nodes(), 1, device=g.device)),
            y_vl,
        ).item()

        return _accuracy_tr, _accuracy_te, _accuracy_vl


    from functools import partial
    if args.stag == "normal":
        dgl.function.sum = partial(stag.stochastic_sum_normal, alpha=args.alpha)

    elif args.stag == "uniform":
        dgl.function.sum = partial(stag.stochastic_sum_uniform, alpha=args.alpha)

    net = Net(
        layer=args.layer,
        in_features=1,
        out_features=10,
        hidden_features=args.hidden_features,
        activation=args.activation,
        depth=args.depth,
    )

    optimizer = torch.optim.Adam(net.parameters(), args.lr)

    ds = dgl.data.CoraGraphDataset()
    g = ds[0]

    if torch.cuda.is_available():
        net = net.to('cuda:0')
        g = g.to('cuda:0')

    accuracy_tr = []
    accuracy_te = []
    accuracy_vl = []

    for idx_epoch in range(args.n_epochs):
        optimizer.zero_grad()
        y_pred = net(g_tr, torch.zeros(g_tr.number_of_nodes(), 1, device=g_tr.device))
        y_true = y_tr
        loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        loss.backward()
        optimizer.step()

        if idx_epoch % args.report_interval == 0:
            _accuracy_tr, _accuracy_te, _accuracy_vl = sampling_performance(g, net, n_samples=1)

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


    _accuracy_tr, _accuracy_te, _accuracy_vl = sampling_performance(g, net, n_samples=16)


    import pandas as pd
    df = pd.DataFrame.from_dict(
        {
            "accuracy_tr": [accuracy_tr[-1]],
            "accuracy_vl": [accuracy_vl[-1]],
            "accuracy_te": [accuracy_te[-1]],
            "sampled_tr": _accuracy_tr,
            "sampled_te": _accuracy_te,
            "sampled_vl": _accuracy_vl,
        }
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
    args = parser.parse_args()
    run(args)
