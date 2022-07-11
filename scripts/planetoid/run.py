import torch
import dgl
from stag.layers import Stag

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = Stag(1433, 8, 8, activation=torch.nn.functional.elu)
        self.layer1 = Stag(64, 7)

    def forward(self, graph, feat):
        feat = self.layer0(graph, feat)
        feat = feat.flatten(-1, -2)
        feat = self.layer1(graph, feat)
        feat = feat.mean(-2)
        return feat

def run():
    from dgl.data import CoraGraphDataset
    g = CoraGraphDataset()[0]
    g = dgl.add_self_loop(g)
    model = Model()

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


    for _ in range(1000):
        model.train()
        optimizer.zero_grad()
        y_hat = model(g, g.ndata["feat"])[g.ndata["train_mask"]]
        y = g.ndata["label"][g.ndata["train_mask"]]
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_hat = model(g, g.ndata["feat"])[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            accuracy = float((y_hat == y).sum()) / len(y_hat)

            if early_stopping([loss, -accuracy], model) is True:
                model.load_state_dict(early_stopping.best_state)
                break

    model.eval()
    with torch.no_grad():
        y_hat = model(g, g.ndata["feat"])[g.ndata["test_mask"]]
        y = g.ndata["label"][g.ndata["test_mask"]]
        accuracy = float((y_hat == y).sum()) / len(y_hat)
        print(accuracy)

if __name__ == "__main__":
    run()
