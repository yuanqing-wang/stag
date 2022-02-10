import torch
import dgl
import stag
from sklearn.metrics import f1_score


def run(args):
    ds_tr = dgl.data.ppi.PPIDataset("train")
    ds_vl = dgl.data.ppi.PPIDataset("valid")
    ds_te = dgl.data.ppi.PPIDataset("test")

    g_vl = dgl.batch(list(ds_vl))
    g_te = dgl.batch(list(ds_te))
    ds_tr_loader = dgl.dataloading.GraphDataLoader(ds_tr, batch_size=2, shuffle=True)

    in_features = 50
    out_features = 121

    layers = torch.nn.ModuleList()

    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                in_features,
                args.hidden_features,
                activation=torch.nn.functional.relu,
            ),
            q_a=torch.distributions.Normal(1.0, args.std),
        )
    )

    for idx in range(1, args.depth-1):
        layers.append(
            stag.layers.StagLayer(
                dgl.nn.GraphConv(
                    args.hidden_features,
                    args.hidden_features,
                    activation=torch.nn.functional.elu,
                ),
                q_a=torch.distributions.Normal(1.0, args.std),
            ),
        )

    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                args.hidden_features,
                out_features,
                activation=torch.nn.functional.sigmoid,
            ),
            q_a=torch.distributions.Normal(1.0, args.std),
        )
    )

    model = stag.models.StagModel(
        layers=layers,
        likelihood=stag.likelihoods.BernoulliLikelihood(),
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g_vl = g_vl.to("cuda:0")
        g_te = g_te.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.1, patience=100)

    for _ in range(args.n_epochs):
        for g in ds_tr_loader:
            if torch.cuda.is_available():
                g = g.to("cuda:0")
            optimizer.zero_grad()
            loss = model.loss(g, g.ndata['feat'], g.ndata['label'])
            loss.backward()
            optimizer.step()

        y_hat = model(g_vl, g_vl.ndata['feat'], return_parameters=True)
        y = g_vl.ndata['label']
        y = y.detach().cpu().flatten()
        y_hat = (y_hat.detach().cpu().flatten() > 0.5) * 1
        f1 = f1_score(y, y_hat)
        scheduler.step(f1)
        if optimizer.param_groups[0]["lr"] <= 0.1 * args.learning_rate: break

    y_hat = model(g_te, g_te.ndata['feat'], return_parameters=True)
    y = g_te.ndata['label']
    y = y.detach().cpu().flatten()
    y_hat = (y_hat.detach().cpu().flatten() > 0.5) * 1
    f1_te = f1_score(y, y_hat)

    performance = {'f1_te': f1_te, 'f1_vl': f1}
    print(performance)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args=parser.parse_args()
    run(args)
