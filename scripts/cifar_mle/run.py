import torch
import dgl
import stag
from torch.utils.data import Dataset, DataLoader

class _Dataset(Dataset):
    def __init__(self, ds_raw):
        super(_Dataset, self).__init__()
        self.ds_raw = ds_raw

    def __getitem__(self, idx):
        h = self.ds_raw[idx]['x']
        x = self.ds_raw[idx]['pos']
        y = torch.tensor(self.ds_raw[idx]['y'])
        g = dgl.graph(
            (
                self.ds_raw[idx]['edge_index'][0],
                self.ds_raw[idx]['edge_index'][1],
            ),
        )

        # g = dgl.remove_self_loop(g)
        # g = dgl.add_self_loop(g)

        g.ndata['h'] = h
        g.ndata['x'] = x
        return g, y

    def __len__(self):
        return len(self.ds_raw)

def collate_fn(batch):
    gs, ys = zip(*batch)
    return dgl.batch(gs), torch.stack(ys, dim=0)

def run(args):
    ds_raw = torch.load("MNIST_v2.pt")
    
    ds_tr, ds_vl, ds_te = ds_raw

    ds_tr_loader = DataLoader(_Dataset(ds_tr), batch_size=args.batch_size, pin_memory=True, collate_fn=collate_fn)
    ds_vl_loader = DataLoader(_Dataset(ds_vl), batch_size=args.batch_size, pin_memory=True, collate_fn=collate_fn)
    ds_te_loader = DataLoader(_Dataset(ds_te), batch_size=args.batch_size, pin_memory=True, collate_fn=collate_fn)

    layers = torch.nn.ModuleList()
    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                3,
                args.hidden_features,
            ),
            q_a=torch.distributions.Normal(1.0, args.std, validate_args=False),
        )
    )

    layers.append(
        stag.layers.FeatOnlyLayer(
            torch.nn.Sequential(
                torch.nn.BatchNorm1d(args.hidden_features),
                torch.nn.ReLU(),
            ),
        ),
    )


    for idx in range(args.depth-2):
        layers.append(
            stag.layers.StagLayer(
                stag.zoo.GCN(
                    args.hidden_features,
                    args.hidden_features,
                ),
            q_a=torch.distributions.Normal(1.0, args.std, validate_args=False),
            ),
        )

        layers.append(
            stag.layers.FeatOnlyLayer(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(args.hidden_features),
                    torch.nn.ReLU(),
                ),
            ),
        )

    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                args.hidden_features,
                args.hidden_features,
            ),
            q_a=torch.distributions.Normal(1.0, args.std, validate_args=False),
        )
    )

    layers.append(
        stag.layers.MeanNodes(),
    )

    layers.append(
        stag.layers.FeatOnlyLayer(
            torch.nn.Sequential(
                torch.nn.Linear(args.hidden_features, 10),
                torch.nn.Softmax(dim=-1),
            )
        )
    )

    model = stag.models.StagModel(
        layers=layers,
        likelihood=stag.likelihoods.CategoricalLikelihood(),
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    for _ in range(1000):
        model.train()
        for g, y in ds_tr_loader:
            if torch.cuda.is_available():
                g = g.to("cuda:0")
                y = y.cuda()

            optimizer.zero_grad()
            loss = model.loss(g, torch.cat([g.ndata['h'], g.ndata['x']], dim=-1), y)

            loss.backward()
            optimizer.step()


        model.eval()
        with torch.no_grad():
            losses = 0.0
            ys = []
            ys_hat = []
            for g, y in ds_vl_loader:
                ys = []
                ys_hat = []
                if torch.cuda.is_available():
                    g = g.to("cuda:0")
                    y = y.cuda()

                loss = model.loss(g, torch.cat([g.ndata['h'], g.ndata['x']], dim=-1), y)
                losses += loss.item()
                ys.append(y)
                y_hat = model(g, torch.cat([g.ndata['h'], g.ndata['x']], dim=-1), return_parameters=True).argmax(dim=-1)
                ys_hat.append(y_hat)


            ys = torch.cat(ys, dim=0).flatten()
            ys_hat = torch.cat(ys_hat, dim=0).flatten()
            accuracy_vl = (ys == ys_hat).sum() / ys_hat.shape[0]
            print(accuracy_vl)

            scheduler.step(losses)
            if optimizer.param_groups[0]['lr'] < 1e-5:
                break

    model.eval()
    with torch.no_grad():
        ys = []
        ys_hat = []
        for g, y in ds_te_loader:
            if torch.cuda.is_available():
                g = g.to("cuda:0")
                y = y.cuda()
            ys.append(y)
            y_hat = model(g, torch.cat([g.ndata['h'], g.ndata['x']], dim=-1), return_parameters=True).argmax(dim=-1)
            ys_hat.append(y_hat)

        ys = torch.cat(ys, dim=0).flatten()
        ys_hat = torch.cat(ys_hat, dim=0).flatten()
        accuracy_te = (ys == ys_hat).sum() / ys_hat.shape[0]


        ys = []
        ys_hat = []
        for g, y in ds_vl_loader:
            if torch.cuda.is_available():
                g = g.to("cuda:0")
                y = y.cuda()
            ys.append(y)
            y_hat = model(g, torch.cat([g.ndata['h'], g.ndata['x']], dim=-1), return_parameters=True).argmax(dim=-1)
            ys_hat.append(y_hat)

        ys = torch.cat(ys, dim=0).flatten()
        ys_hat = torch.cat(ys_hat, dim=0).flatten()
        accuracy_te = (ys == ys_hat).sum() / ys_hat.shape[0]
    
    performance = {"accuracy_te": accuracy_te, "accuracy_vl": accuracy_vl}
    import json
    with open(args.out + ".json", "w") as file_handle:
        json.dump(performance, file_handle)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--std", type=float, default=0.0)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()
    run(args)
