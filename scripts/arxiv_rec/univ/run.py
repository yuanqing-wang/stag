import torch
import dgl
import stag

def run(args):
    if args.data == "cora":
        ds = dgl.data.CoraGraphDataset()
        g = ds[0]
        in_features = 1433
        out_features = 7
        p = 1

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

    elif args.data == "reddit":
        ds = dgl.data.RedditDataset()
        g = ds[0]
        in_features = 602
        out_features = 41
        p = 1

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    g.ndata['feat'] = torch.nn.functional.normalize(
        g.ndata['feat'],
        dim=-1,
        p=p,
    )

    model = getattr(stag.zoo, args.model)

    if args.distribution == "Normal":
        q_a = torch.distributions.Normal(1.0, args.std, validate_args=False)
        norm = False

    elif args.distribution == "Uniform":
        import math
        half_range = args.std * math.sqrt(3.0)
        q_a = torch.distributions.Uniform(1.0-half_range, 1.0+half_range, validate_args=False)
        norm = False

    elif args.distribution == "Bernoulli":
        import math
        prob = 0.5 * (1.0 +  math.sqrt(1 - 4.0 * args.std ** 2))
        q_a = torch.distributions.Bernoulli(probs=prob)
        norm = True

    layers = torch.nn.ModuleList()

    layers.append(
        stag.layers.StagLayer(
            model(
                in_features,
                args.hidden_features,
                activation=torch.nn.functional.relu,
            ),
            q_a=q_a,
            norm=norm,
        )
    )

    layers.append(
        stag.layers.StagLayer(
            model(
                args.hidden_features,
                out_features,
                activation=lambda x: torch.nn.functional.softmax(x, dim=-1),
            ),
            q_a=q_a,
            norm=norm,
        )
    )

    model = stag.models.StagModel(
        layers=layers,
    )

    print(model)

    if torch.cuda.is_available():
        model = model.cuda()# .to("cuda:0")
        g = g.to("cuda:0")

    layer0, layer1 = [layer for layer in model.layers if isinstance(layer, stag.layers.StagLayer)]

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    losses = []
    for idx_epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        
        loss = model.loss(g, g.ndata["feat"], y=g.ndata["label"], mask=g.ndata["train_mask"], n_samples=args.n_samples_training)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_vl = model.loss(
                 g, g.ndata['feat'], y=g.ndata['label'], mask=g.ndata["val_mask"],
                 n_samples=args.n_samples,
            )

            scheduler.step(loss_vl)
            if optimizer.param_groups[0]['lr'] < 1e-6:
                break


    model.eval()
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
    parser.add_argument("--distribution", type=str, default="Normal")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--n_samples_training", type=int, default=2)
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--std", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    args=parser.parse_args()
    run(args)
