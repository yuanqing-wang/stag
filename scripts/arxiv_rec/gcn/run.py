import numpy as np
import torch
import dgl
import stag

def run(args):
    if args.data == "cora":
        ds = dgl.data.CoraGraphDataset()
        g = ds[0]
        in_features = 1433
        out_features = 7

    elif args.data == "citeseer":
        ds = dgl.data.CiteseerGraphDataset()
        g = ds[0]
        in_features = 3703
        out_features = 6

    elif args.data == "pubmed":
        ds = dgl.data.PubmedGraphDataset()
        g = ds[0]
        in_features = 500
        out_features = 3

    elif args.data == "reddit":
        ds = dgl.data.RedditDataset()
        g = ds[0]
        in_features = 602
        out_features = 41

    elif args.data == "arxiv":
        from ogb.nodeproppred import DglNodePropPredDataset
        ds = DglNodePropPredDataset("ogbn-arxiv")
        g, y = ds[0]
        g.ndata['label'] = y.flatten()
        split_idx = ds.get_idx_split()

        train_mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        valid_mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool)

        train_mask[split_idx["train"]] = True
        test_mask[split_idx["test"]] = True
        valid_mask[split_idx["valid"]] = True

        g.ndata["train_mask"] = train_mask
        g.ndata["test_mask"] = test_mask
        g.ndata["val_mask"] = valid_mask

        in_features = 128
        out_features = 40

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = dgl.add_reverse_edges(g)


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
    p_a = q_a

    layers.append(
        stag.layers.StagLayer(
            model(
                in_features,
                args.hidden_features,
            ),
            q_a=stag.distributions.AmortizedDistribution(in_features, 1, init_like=p_a),
            norm=norm,
        )
    )

    layers.append(
        stag.layers.FeatOnlyLayer(
            torch.nn.Sequential(
                torch.nn.BatchNorm1d(args.hidden_features),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
            ),
        ),
    )

    for _ in range(args.depth - 2):
        layers.append(
            stag.layers.StagLayer(
                model(
                    args.hidden_features,
                    args.hidden_features,
                ),
                q_a=stag.distributions.AmortizedDistribution(args.hidden_features, 1, init_like=p_a),
                p_a=p_a,
                norm=norm,
            )
        )

        layers.append(
            stag.layers.FeatOnlyLayer(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(args.hidden_features),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                ),
            ),
        )


    layers.append(
        stag.layers.StagLayer(
            model(
                args.hidden_features,
                out_features,
                activation=lambda x: torch.nn.functional.softmax(x, dim=-1),
            ),
            q_a=stag.distributions.AmortizedDistribution(args.hidden_features, 1, init_like=p_a),
            p_a=p_a,
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

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    accuracy_vl_array = []
    accuracy_te_array = []

    for idx_epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        
        loss = model.loss(g, g.ndata["feat"], y=g.ndata["label"], mask=g.ndata["train_mask"], n_samples=args.n_samples_training)

        loss.backward()
        optimizer.step()

        
        model.eval()
        with torch.no_grad():
            y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True).argmax(dim=-1)[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy_vl = float((y_hat == y).sum()) / len(y_hat)

            y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True).argmax(dim=-1)[g.ndata["test_mask"]]
            y = g.ndata["label"][g.ndata["test_mask"]]
            accuracy_te = float((y_hat == y).sum()) / len(y_hat)
            
            accuracy_vl_array.append(accuracy_vl)
            accuracy_te_array.append(accuracy_te)

        best_epoch = np.array(accuracy_vl_array).argmax()
        accuracy_vl = accuracy_vl_array[best_epoch]
        accuracy_te = accuracy_te_array[best_epoch]

    
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
    parser.add_argument("--n_samples_training", type=int, default=1)
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--std", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args=parser.parse_args()
    run(args)
