import numpy as np
import torch
import dgl
import stag

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
        y_pred = net(g, feats, n_samples=n_samples)
        y_true = labels

        _accuracy_tr = evaluator.eval({"y_true": y_true[train_idx], "y_pred": y_pred[train_idx]})["acc"]
        _accuracy_te = evaluator.eval({"y_true": y_true[test_idx], "y_pred": y_pred[test_idx]})["acc"]
        _accuracy_vl = evaluator.eval({"y_true": y_true[valid_idx], "y_pred": y_pred[valid_idx]})["acc"]

        return _accuracy_tr, _accuracy_te, _accuracy_vl

    layer = dgl.nn.GraphConv

    net = stag.vi.StagVI_NodeClassification_RC(
        layer=layer,
        in_features=feats.size(-1),
        out_features=dataset.num_classes,
        hidden_features=args.hidden_features,
        activation=args.activation,
        depth=args.depth,
        kl_scaling=args.kl_scaling,
        a_prior=args.a_prior,
        a_mu_init_std=args.a_mu_init_std,
        a_log_sigma_init=args.a_log_sigma_init,
    )

    if torch.cuda.is_available():
        net = net.cuda()
        g = g.to("cuda:0")
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()
        valid_idx = valid_idx.cuda()
        feats = feats.cuda()
        labels = labels.cuda()

    import itertools
    optimizer = torch.optim.Adam(
        [
            {
                "params": itertools.chain(
                    *[
                        getattr(net, "gn%s" % idx).parameters()
                        for idx in range(net.depth-1)
                    ]
                ),
                "lr": 1e-3,
                "weight_decay": 5e-4,
            },
            {
                "params": getattr(net, "gn%s" % (net.depth-1)).parameters(),
                "lr": 1e-3,
            },
            {
                "params": [getattr(net, "a_mu_%s" % idx) for idx in range(net.depth)] +\
                    [getattr(net, "a_log_sigma_%s" % idx) for idx in range(net.depth)],
                "lr": 0.1,
            },

        ]
    )

    accuracy_tr = []
    accuracy_te = []
    accuracy_vl = []

    net.train()

    for idx_epoch in range(args.n_epochs):
        optimizer.zero_grad()
        loss = net.loss(g, feats, mask=train_idx, n_samples=8)
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
