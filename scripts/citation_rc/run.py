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
        stag.layers.StagMeanFieldVariationalInferenceLayer(
            dgl.nn.GraphConv(
                in_features,
                args.hidden_features,
                activation=torch.nn.functional.relu,
            ),
        )
    )
    for idx in range(1, args.depth-1):
        layers.append(
            stag.layers.StagMeanFieldVariationalInferenceLayer(
                dgl.nn.GraphConv(
                    args.hidden_features,
                    args.hidden_features,
                    activation=torch.nn.functional.relu,
                ),
            ),
        )
    layers.append(
        stag.layers.StagMeanFieldVariationalInferenceLayer(
            dgl.nn.GraphConv(
                args.hidden_features,
                out_features,
                activation=lambda x: torch.nn.functional.softmax(x, dim=-1),
            ),
        )
    )

    model = stag.models.StagModel(
        layers=layers,
        kl_scaling=1e-3,
    )

    if torch.cuda.is_available():
        model = model.to("cuda:0")
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    from stag.utils import EarlyStopping
    early_stopping = EarlyStopping()

    for idx_epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(
            g, g.ndata['feat'], y=g.ndata['label'], mask=g.ndata["train_mask"],
            n_samples=args.n_samples,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_vl = model.loss(
                g, g.ndata['feat'], y=g.ndata['label'], mask=g.ndata["val_mask"],
                n_samples=args.n_samples,
                kl_scaling=0.0,
            )

            y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True).argmax(dim=-1)[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy_vl = float((y_hat == y).sum()) / len(y_hat)

            print(accuracy_vl)

            if early_stopping(loss_vl): break

    y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True).argmax(dim=-1)[g.ndata["val_mask"]]
    y = g.ndata["label"][g.ndata["val_mask"]]
    accuracy_vl = float((y_hat == y).sum()) / len(y_hat)

    y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True).argmax(dim=-1)[g.ndata["test_mask"]]
    y = g.ndata["label"][g.ndata["test_mask"]]
    accuracy_te = float((y_hat == y).sum()) / len(y_hat)


    import pandas as pd
    df = pd.DataFrame(
        {
            args.data: {
                "accuracy_te": accuracy_te,
                "accuracy_vl": accuracy_vl,
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
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=4)
    args=parser.parse_args()
    run(args)
