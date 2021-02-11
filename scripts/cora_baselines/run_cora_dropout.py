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
        
    def forward(self, g, x):
        # ensure local scope
        g = g.local_var()

        x = self.gn0(g, x)
        x = torch.nn.Dropout(args.alpha)(x)

        for idx in range(1, self.depth-1):
            x = getattr(self, "gn%s" % idx)(g, x)
            x = torch.nn.Dropout(args.alpha)(x)
        
        if self.depth > 1:
            x = getattr(self, "gn%s" % (self.depth-1))(g, x)
            x = torch.nn.Dropout(args.alpha)(x)

        return x

def accuracy_between(y_pred, y_true):
    if y_pred.dim() >= 2:
        y_pred = y_pred.argmax(dim=-1)
    return (y_pred == y_true).sum() / y_pred.shape[0]

def sampling_performance(g, net, n_samples=16):
    y_pred = torch.stack([net(g, g.ndata['feat']).detach() for _ in range(n_samples)], dim=0).mean(dim=0)
    y_true = g.ndata['label']

    _accuracy_tr = accuracy_between(y_pred[g.ndata['train_mask']], y_true[g.ndata['train_mask']]).item()
    _accuracy_te = accuracy_between(y_pred[g.ndata['test_mask']], y_true[g.ndata['test_mask']]).item()
    _accuracy_vl = accuracy_between(y_pred[g.ndata['val_mask']], y_true[g.ndata['val_mask']]).item()

    return _accuracy_tr, _accuracy_te, _accuracy_vl

def run(args):
    from functools import partial
    from dgl.nn import pytorch as dgl_nn

    if args.stag.startswith("none") == False:
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

        layer = lambda in_feats, out_feats, activation: dgl_nn.GINConv(
            apply_func=Layer(in_feats, out_feats, activation=activation),
            aggregator_type="sum",
        )

    else:
        layer = getattr(dgl_nn, args.layer)


    net = Net(
        layer=layer,
        in_features=1433,
        out_features=7,
        hidden_features=args.hidden_features,
        activation=args.activation,
        depth=args.depth,
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
            }
        ]

    )
    ds = dgl.data.CoraGraphDataset()
    g = ds[0]

    if torch.cuda.is_available():
        net = net.to('cuda:0')
        g = g.to('cuda:0')

    accuracy_tr = []
    accuracy_te = []
    accuracy_vl = []

    for idx_epoch in range(args.n_epochs):
        net.train()
        optimizer.zero_grad()
        y_pred = net(g, g.ndata['feat'])[g.ndata['train_mask']]
        y_true = g.ndata['label'][g.ndata['train_mask']]

        loss = torch.nn.functional.nll_loss(y_pred.log_softmax(dim=-1), y_true)
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
    args = parser.parse_args()
    run(args)
