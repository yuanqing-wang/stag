import torch
import dgl
import stag

def run(args):
    if args.data == "cora":
        ds = dgl.data.CoraGraphDataset(reverse_edge=False)
        g = ds[0]
        in_features = 1433
        out_features = 7
        num_heads = 1

    elif args.data == "citeseer":
        ds = dgl.data.CiteseerGraphDataset(reverse_edge=False)
        g = ds[0]
        in_features = 3703
        out_features = 6
        num_heads = 1

    elif args.data == "pubmed":
        ds = dgl.data.PubmedGraphDataset(reverse_edge=False)
        g = ds[0]
        in_features = 500
        out_features = 3
        # args.learning_rate = 0.01
        # args.weight_decay = 0.001
        num_heads = 8

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    layers = torch.nn.ModuleList()
    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GAT(
                in_features,
                8,
                num_heads=8,
                feat_drop=0.6,
                attn_drop=0.6,
                activation=torch.nn.functional.elu,
            ),
            q_a=torch.distributions.Normal(1.0, args.std, validate_args=False),
        )
    )


    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GAT(
                64,
                out_features,
                num_heads=num_heads,
                last=True,
                feat_drop=0.6,
                attn_drop=0.6,
                activation=lambda x: torch.nn.functional.softmax(x, dim=-1),
            ),
            q_a=torch.distributions.Normal(1.0, args.std, validate_args=False),
        )
    )

    model = stag.models.StagModel(
        layers=layers,
    )

    print(model)

    weights = []
    others = []

    for name, parameter in model.named_parameters():
        if "weight" in name:
            weights.append(parameter)
        else:
            others.append(parameter)

    if torch.cuda.is_available():
        model = model.cuda()# .to("cuda:0")
        g = g.to("cuda:0")


    optimizer = torch.optim.Adam(
        [
            {'params': weights, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
            {'params': others, 'lr': args.learning_rate, 'weight_decay': 0.0},
        ],
    )

    early_stopping = stag.utils.EarlyStopping(patience=100)

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
            ).item()

            y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True)[g.ndata["val_mask"]].argmax(dim=-1)
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy_vl = float((y_hat == y).sum()) / len(y_hat)

            if early_stopping([loss_vl, -accuracy_vl], model) is True:
                model.load_state_dict(early_stopping.best_state)
                break

    model.eval()
    with torch.no_grad():
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
    parser.add_argument("--model", type=str, default="GAT")
    parser.add_argument("--n_samples_training", type=int, default=2)
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    args=parser.parse_args()
    run(args)
