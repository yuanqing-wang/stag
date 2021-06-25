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
            )
        )
    )
    for idx in range(1, args.depth-1):
        layers.append(
            stag.layers.StagLayer(
                dgl.nn.GraphConv(
                    args.hidden_features,
                    args.hidden_features,
                    activation=torch.nn.functional.relu,
                )
            )
        )
    layers.append(
        stag.layers.StagLayer(
            dgl.nn.GraphConv(
                args.hidden_features,
                out_features,
                activation=lambda x: torch.nn.functional.softmax(x, dim=-1),
            )
        )
    )

    model = stag.models.StagModel(
        layers=layers,
    )

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = []
    from stag.utils import EarlyStopping
    early_stopping = EarlyStopping()
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

        y_hat = model(g, g.ndata["feat"], n_samples=4)[g.ndata["test_mask"]]
        y = g.ndata["label"][g.ndata["test_mask"]]
        accuracy_te = float((y_hat == y).sum()) / len(y_hat)
        print(accuracy_te)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=10)
    args=parser.parse_args()
    run(args)
