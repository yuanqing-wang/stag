import torch
import dgl
import stag

def run(args):
    from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
    from torch.utils.data import DataLoader

    dataset = DglGraphPropPredDataset(name="ogbg-molhiv")
    in_features = 9
    out_features = 1

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)

    layers = torch.nn.ModuleList()
    layers.append(
        stag.layers.StagLayer(
            dgl.nn.GraphConv(
                in_features,
                args.hidden_features,
                activation=torch.nn.functional.relu,
            ),
            edge_weight_distribution=torch.distributions.Normal(1.0, args.std),
        )
    )

    layers.append(
        stag.layers.FeatOnlyLayer(
            torch.nn.Dropout(),
        ),
    )

    for idx in range(1, args.depth-1):
        layers.append(
            stag.layers.StagLayer(
                dgl.nn.GraphConv(
                    args.hidden_features,
                    args.hidden_features,
                    activation=torch.nn.functional.relu,
                ),
                edge_weight_distribution=torch.distributions.Normal(1.0, args.std),
            ),
        )

        layers.append(
            stag.layers.FeatOnlyLayer(
                torch.nn.Dropout(),
            ),
        )

    layers.append(
        stag.layers.StagLayer(
            dgl.nn.GraphConv(
                args.hidden_features,
                args.hidden_features,
            ),
            edge_weight_distribution=torch.distributions.Normal(1.0, args.std),
        )
    )

    layers.append(
        stag.layers.SumNodes(),
    )

    # layers.append(
    #     stag.layers.FeatOnlyLayer(
    #         torch.nn.Sequential(
    #             torch.nn.Linear(args.hidden_features, args.hidden_features),
    #             torch.nn.ReLU(),
    #             torch.nn.Linear(args.hidden_features, out_features),
    #         )
    #     )
    # )

    model = stag.models.StagModel(
        layers=layers,
    )

    if torch.cuda.is_available():
        model = model.to("cuda:0")
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=5e-4)
    from stag.utils import EarlyStopping
    early_stopping = EarlyStopping(patience=10)

    for idx_epoch in range(args.n_epochs):
        model.train()
        for g, y in train_loader:
            optimizer.zero_grad()
            y_hat = model.forward(g, g.ndata["feat"], n_samples=1, return_parameters=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_vl = model.loss(
                g, g.ndata['feat'], y=g.ndata['label'], mask=g.ndata["val_mask"],
                n_samples=args.n_samples,
            )

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
