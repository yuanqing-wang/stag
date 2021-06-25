import torch
import dgl
import stag

def run(args):
    if args.data == "cora":
        ds = dgl.data.CoraGraphDataset()
        g = ds[0]
        in_features = 1433
        out_features = 7

    layers = torch.nn.ModuleList()
    layers.append(
        stag.layers.StagLayer(
            dgl.nn.GraphConv(
                in_features,
                args.hidden_features,
                activation=torch.nn.functional.relu,
            ),
            edge_weight_distribution=torch.distributions.Normal(1.0, 0.8),
        )
    )
    for idx in range(1, args.depth-1):
        layers.append(
            stag.layers.StagLayer(
                dgl.nn.GraphConv(
                    args.hidden_features,
                    args.hidden_features,
                    activation=torch.nn.functional.relu,
                ),
                edge_weight_distribution=torch.distributions.Normal(1.0, 0.8),
            ),
        )
    layers.append(
        stag.layers.StagLayer(
            dgl.nn.GraphConv(
                args.hidden_features,
                out_features,
                activation=lambda x: torch.nn.functional.softmax(x, dim=-1),
            ),
            edge_weight_distribution=torch.distributions.Normal(1.0, 0.8),
        )
    )

    model = stag.models.StagModel(
        layers=layers,
    )

    if torch.cuda.is_available():
        model = model.to("cuda:0")
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    accuracy_vl = []
    accuracy_te = []

    for idx_epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(
            g, g.ndata['feat'], y=g.ndata['label'], mask=g.ndata["train_mask"],
            n_samples=4,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_vl = model.loss(
                g, g.ndata['feat'], y=g.ndata['label'], mask=g.ndata["val_mask"],
                n_samples=4,
            )

            y_hat = model(g, g.ndata["feat"], n_samples=4)[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            _accuracy_vl = float((y_hat == y).sum()) / len(y_hat)

            y_hat = model(g, g.ndata["feat"], n_samples=4)[g.ndata["test_mask"]]
            y = g.ndata["label"][g.ndata["test_mask"]]
            _accuracy_te = float((y_hat == y).sum()) / len(y_hat)

            accuracy_vl.append(_accuracy_vl)
            accuracy_te.append(_accuracy_te)

    import numpy as np
    best_epoch = np.array(accuracy_vl).argmax()

    import pandas as pd
    df = pd.DataFrame(
        {
            args.data: {
                "accuracy_te": accuracy_te[best_epoch],
                "accuracy_vl": accuracy_vl[best_epoch],
            }
        }
    )

    df.to_csv("%s.csv" % args.out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--out", type=str, default="out")
    args=parser.parse_args()
    run(args)
