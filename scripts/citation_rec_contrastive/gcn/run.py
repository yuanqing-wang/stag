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

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    g.ndata['feat'] = torch.nn.functional.normalize(
        g.ndata['feat'],
        dim=-1,
        p=p,
    )

    kl_scaling = args.kl_scaling * g.number_of_edges() * g.ndata["train_mask"].sum() / (g.ndata["train_mask"].shape[0] ** 2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    '''
    p_a = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(torch.ones(2, device=device)),
        component_distribution=torch.distributions.Normal(
            torch.tensor([0.0, 1.0], device=device),
            torch.tensor([args.std, args.std], device=device),
        ),
    )
    '''

    p_a = torch.distributions.Normal(0.5, args.std)

    layers = torch.nn.ModuleList()

    if True: # float(args.std) == 0.0:
        layers.append(
            stag.layers.FeatOnlyLayer(
                torch.nn.Dropout(0.5),
            ),
        )

    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                in_features,
                args.hidden_features,
                activation=torch.nn.functional.relu,
            ),
            q_a=stag.distributions.AmortizedDistribution(in_features, 1),
            p_a=p_a,
            vi=True,
            # norm=True,
        )
    )

    if True: # float(args.std) == 0.0:
        layers.append(
            stag.layers.FeatOnlyLayer(
                torch.nn.Dropout(0.5),
            ),
        )


    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                args.hidden_features,
                out_features,
                activation=lambda x: torch.nn.functional.softmax(x, dim=-1),
            ),
            q_a=stag.distributions.AmortizedDistribution(args.hidden_features, 1),
            p_a=p_a,
            vi=True,
            # norm=True,
        )
    )

    model = stag.models.StagModelContrastive(
        layers=layers,
        kl_scaling=kl_scaling,
    )

    print(model)
    if torch.cuda.is_available():
        model = model.cuda()# .to("cuda:0")
        g = g.to("cuda:0")

    layer0, layer1 = [layer for layer in model.layers if isinstance(layer, stag.layers.StagLayer)]

    optimizer_nn = torch.optim.Adam(
        [
            {'params': layer0.base_layer.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
            {'params': layer1.base_layer.parameters(), 'lr': args.learning_rate},
        ]
   
    )

    optimizer_qa = torch.optim.Adam(
         [
            {'params': layer0.q_a.parameters(), 'lr': args.learning_rate, 'weight_decay': 0.0},
            {'params': layer1.q_a.parameters(), 'lr': args.learning_rate},
        ]
          
    )

    early_stopping = stag.utils.EarlyStopping(patience=10)

    for idx_epoch in range(500):
        model.train()
        optimizer_nn.zero_grad()
        optimizer_qa.zero_grad()

        nll, reg = model.loss_terms(g, g.ndata["feat"], y=g.ndata["label"], mask=g.ndata["train_mask"], n_samples=args.n_samples_training)
        reg.backward()
        print(reg)
        print(layer0.q_a.mean.mean(), layer1.q_a.mean.mean())
        
        optimizer_nn.step()
        optimizer_qa.step()


    losses = []
    for idx_epoch in range(args.n_epochs):
        model.train()
        optimizer_nn.zero_grad()
        optimizer_qa.zero_grad()

        nll, reg = model.loss_terms(g, g.ndata["feat"], y=g.ndata["label"], mask=g.ndata["train_mask"], n_samples=args.n_samples_training)

        (nll+reg).backward(
            inputs=list(layer0.q_a.parameters()) + list(layer1.q_a.parameters()),
            retain_graph=True,
        )

        (nll+reg).backward(
            inputs=list(layer0.base_layer.parameters()) + list(layer1.base_layer.parameters())
        )

        optimizer_nn.step()
        optimizer_qa.step()

        model.eval()
        with torch.no_grad():
            loss_vl = model.loss(
                 g, g.ndata['feat'], y=g.ndata['label'], mask=g.ndata["val_mask"],
                 n_samples=args.n_samples,
            )

            y_hat = model.forward(g, g.ndata["feat"], n_samples=args.n_samples, return_parameters=True).argmax(dim=-1)[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy_vl = float((y_hat == y).sum()) / len(y_hat)

            if early_stopping([loss_vl, -accuracy_vl], model) is True:
                model.load_state_dict(early_stopping.best_state)
                break


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
