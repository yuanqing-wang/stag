import numpy as np
import torch
import dgl
import stag

def accuracy_between(y_pred, y_true):
    if y_pred.dim() >= 2:
        y_pred = y_pred.argmax(dim=-1)
    return (y_pred == y_true).sum() / y_pred.shape[0]

def sampling_performance(g, net, n_samples=16):
    y_pred = net(g, g.ndata['feat'], n_samples=n_samples)
    y_true = g.ndata['label']

    _accuracy_tr = accuracy_between(y_pred[g.ndata['train_mask']], y_true[g.ndata['train_mask']]).item()
    _accuracy_te = accuracy_between(y_pred[g.ndata['test_mask']], y_true[g.ndata['test_mask']]).item()
    _accuracy_vl = accuracy_between(y_pred[g.ndata['val_mask']], y_true[g.ndata['val_mask']]).item()

    return _accuracy_tr, _accuracy_te, _accuracy_vl

def run(args):
    from functools import partial
    from dgl.nn import pytorch as dgl_nn

    dgl.function.copy_src = dgl.function.copy_u = stag.stag_copy_src_vi
    layer = getattr(dgl_nn, args.layer)

    if args.data == "cora":
        ds = dgl.data.CoraGraphDataset()
        g = ds[0]
        net = stag.vi.StagVI_NodeClassification_RE(
            layer=layer,
            in_features=1433,
            out_features=7,
            hidden_features=args.hidden_features,
            activation=args.activation,
            depth=args.depth,
        )


    elif args.data == "citeseer":
        ds = dgl.data.CiteseerGraphDataset()
        g = ds[0]
        net = stag.vi.StagVI_NodeClassification_RE(
            layer=layer,
            in_features=3703,
            out_features=6,
            hidden_features=args.hidden_features,
            activation=args.activation,
            depth=args.depth,
        )

    import itertools
    optimizer = torch.optim.Adam(net.parameters(), args.lr)

    if torch.cuda.is_available():
        net = net.to('cuda:0')
        g = g.to('cuda:0')

    accuracy_tr = []
    accuracy_te = []
    accuracy_vl = []


    for idx_epoch in range(args.n_epochs):
        net.train()
        optimizer.zero_grad()
        loss = net.loss(g, g.ndata['feat'], g.ndata['label'], mask=g.ndata['train_mask'], n_samples=8)
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


    def find_best_epoch(accuracy_vl):
        for idx in range(len(accuracy_vl)):
            ok = True
            for _idx in range(idx, idx+10):
                if accuracy_vl[_idx] > accuracy_vl[idx]:
                    ok = False
                    break
                if ok is True:
                    return idx
        return len(accuracy_vl) - 1

    best_epoch = find_best_epoch(accuracy_vl)
    import pandas as pd
    df = pd.DataFrame.from_dict(
        {
            "tr": [accuracy_tr[best_epoch]],
            "te": [accuracy_te[best_epoch]],
            "vl": [accuracy_vl[best_epoch]],
        },
    )

    df.to_markdown(open(args.out + "/overview.md", "w"))


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
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
