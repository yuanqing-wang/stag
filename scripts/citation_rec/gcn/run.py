import torch
import dgl
import stag
from itertools import chain

def run(args):
    if args.data == "cora":
        ds = dgl.data.CoraGraphDataset()
        g = ds[0]
        in_features = 1433
        out_features = 7
        p = 2

    elif args.data == "citeseer":
        ds = dgl.data.CiteseerGraphDataset()
        g = ds[0]
        in_features = 3703
        out_features = 6
        p = 1

    elif args.data == "pubmed":
        ds = dgl.data.PubmedGraphDataset()
        g = ds[0]
        in_features = 500
        out_features = 3
        p = 1

    g = dgl.add_self_loop(g)

    g.ndata['feat'] = torch.nn.functional.normalize(
        g.ndata['feat'],
        dim=-1,
        p=p,
    )

    kl_scaling = g.number_of_edges() * g.ndata["train_mask"].sum() / g.ndata["train_mask"].shape[0]

    p_a = torch.distributions.Normal(1.0, args.std)

    layers = torch.nn.ModuleList()
    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                in_features,
                args.hidden_features,
                activation=torch.nn.functional.relu,
            ),
            q_a=stag.distributions.AmortizedDistribution(in_features, in_features, init_like=p_a),
            p_a=p_a,
            vi=True,
        )
    )

    for idx in range(1, args.depth-1):
        layers.append(
            stag.layers.StagLayer(
                stag.zoo.GCN(
                    args.hidden_features,
                    args.hidden_features,
                    activation=torch.nn.functional.relu,
                ),
                q_a=stag.distributions.AmortizedDistribution(args.hidden_features, args.hidden_features, init_like=p_a),
                p_a=p_a,
                vi=True,
            ),
        )

    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                args.hidden_features,
                out_features,
                activation=lambda x: torch.nn.functional.softmax(x, dim=-1),
            ),
            q_a=stag.distributions.AmortizedDistribution(args.hidden_features, args.hidden_features, init_like=p_a),
            p_a=p_a,
            vi=True,
        )
    )

    model = stag.models.StagModel(
        layers=layers,
        kl_scaling=kl_scaling,
    )

    if torch.cuda.is_available():
        model = model.cuda()# .to("cuda:0")
        g = g.to("cuda:0")

    optimizer_nn = torch.optim.Adam(
        chain(*[layer.base_layer.parameters() for layer in model.layers]),
        args.learning_rate,
        weight_decay=args.weight_decay,
    )

    optimizer_qa = torch.optim.Adam(
        chain(*[layer.q_a.parameters() for layer in model.layers]),
        args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nn, "min", factor=0.5, patience=10)

    losses = []
    for idx_epoch in range(args.n_epochs):
        model.train()
        optimizer_nn.zero_grad()
        optimizer_qa.zero_grad()

        nll, reg = model.loss_terms(g, g.ndata["feat"], y=g.ndata["label"], mask=g.ndata["train_mask"], n_samples=args.n_samples_training)

        (nll+reg).backward(
            inputs=list(chain(*[layer.base_layer.parameters() for layer in model.layers])),
            retain_graph=True,
        )

        reg.backward(
            inputs=list(chain(*[layer.q_a.parameters() for layer in model.layers])),
        )

        optimizer_nn.step()
        optimizer_qa.step()

        model.eval()
        with torch.no_grad():
            loss_vl = model.loss(
                 g, g.ndata['feat'], y=g.ndata['label'], mask=g.ndata["val_mask"],
                 n_samples=args.n_samples,
                 # kl_scaling=0.0,
            )
            # y_hat = model.forward(g, g.ndata["feat"], return_parameters=True)[g.ndata["val_mask"]]
            # y = g.ndata["label"][g.ndata["val_mask"]]
            # loss_vl = torch.nn.CrossEntropyLoss()(y_hat, y)

            losses.append(loss_vl.item())

            scheduler.step(loss_vl)

            if optimizer_nn.param_groups[0]["lr"] <= 0.0001 * args.learning_rate: break

    y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True).argmax(dim=-1)[g.ndata["val_mask"]]
    y = g.ndata["label"][g.ndata["val_mask"]]
    accuracy_vl = float((y_hat == y).sum()) / len(y_hat)

    y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True).argmax(dim=-1)[g.ndata["test_mask"]]
    y = g.ndata["label"][g.ndata["test_mask"]]
    accuracy_te = float((y_hat == y).sum()) / len(y_hat)


    performance = {"accuracy_te": accuracy_te, "accuracy_vl": accuracy_vl}
    import json
    with open(args.out + ".json", "w") as file_handle:
        json.dump(performance, file_handle)

    # from matplotlib import pyplot as plt
    # plt.plot(losses)
    # plt.xlabel("t")
    # plt.ylabel("loss")
    # plt.savefig("loss.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples_training", type=int, default=1)
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_scaling", type=float, default=1.0)
    args=parser.parse_args()
    run(args)
