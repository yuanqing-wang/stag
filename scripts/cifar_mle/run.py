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
        g.ndata['h'] = h
        g.ndata['x'] = x
        return g, y

    def __len__(self):
        return len(self.ds_raw)

def collate_fn(batch):
    gs, ys = zip(*batch)
    return dgl.batch(gs), torch.stack(ys, dim=0)

def run():
    ds_raw = torch.load("MNISTSuperpixels.pt")
    ds_tr = ds_raw[0]
    ds_vl = ds_raw[1]

    ds_tr_loader = DataLoader(_Dataset(ds_tr), batch_size=32, pin_memory=True, collate_fn=collate_fn)
    ds_vl_loader = DataLoader(_Dataset(ds_vl), batch_size=32, pin_memory=True, collate_fn=collate_fn)

    layers = torch.nn.ModuleList()
    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                3,
                128,
                activation=torch.nn.functional.elu,
            ),
            q_a=torch.distributions.Normal(0.0, 1.0),
        )
    )

    for idx in range(1, 3):
        layers.append(
            stag.layers.StagLayer(
                stag.zoo.GCN(
                    128,
                    128,
                    activation=torch.nn.functional.elu,
                ),
            q_a=torch.distributions.Normal(0.0, 1.0),
            ),
        )

    layers.append(
        stag.layers.StagLayer(
            stag.zoo.GCN(
                128,
                128,
                activation=torch.nn.functional.elu,
            ),
            q_a=torch.distributions.Normal(0.0, 1.0),
        )
    )

    layers.append(
        stag.layers.SumNodes(),
    )

    layers.append(
        stag.layers.FeatOnlyLayer(
            torch.nn.Sequential(
                torch.nn.Linear(128, 10),
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

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

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
                if torch.cuda.is_available():
                    g = g.to("cuda:0")
                    y = y.cuda()
                loss = model.loss(g, torch.cat([g.ndata['h'], g.ndata['x']], dim=-1), y)
                losses += loss.item()

                ys.append(y)
                y_hat = model(g, torch.cat([g.ndata['h'], g.ndata['x']], dim=-1))
                ys_hat.append(y_hat)

            ys = torch.cat(ys, dim=0)
            ys_hat = torch.cat(ys_hat, dim=0)
            accuracy_vl = (ys == ys_hat).sum() / ys_hat.shape[0]
            print(accuracy_vl)
        scheduler.step(losses)


if __name__ == "__main__":
    run()
