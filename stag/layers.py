from typing import Optional, Callable
import torch
import dgl

class Stag(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        num_samples: int = 1,
        negative_slop: float = 0.2,
        activation: Optional[Callable] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features * num_heads, bias=False)
        self.posterior_parameters = torch.nn.Linear(out_features, 4)

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(num_heads, out_features),
            )
        else:
            self.bias = None

        self.leaky_relu = torch.nn.LeakyReLU(negative_slop)
        self.num_heads = num_heads
        self.num_samples = num_samples
        self.out_features = out_features
        self.activation = activation

    def forward(self, graph, feat):
        graph = graph.local_var()
        feat = self.fc(feat)
        feat = feat.reshape(feat.shape[:-1] + (self.num_heads, self.out_features))
        loc_l, loc_r, log_scale_l, log_scale_r = self.posterior_parameters(feat).split(1, dim=-1)
        graph.ndata["loc_l"] = loc_l
        graph.ndata["loc_r"] = loc_r
        graph.ndata["log_scale_l"] = log_scale_l
        graph.ndata["log_scale_r"] = log_scale_r
        graph.apply_edges(dgl.function.u_add_v("loc_l", "loc_r", "loc"))
        graph.apply_edges(dgl.function.u_add_v("log_scale_l", "log_scale_r", "log_scale"))
        loc = graph.edata["loc"]
        scale = graph.edata["log_scale"].exp()
        loc = self.leaky_relu(loc)
        e = torch.distributions.Normal(loc, scale).rsample((self.num_samples,)).swapaxes(0, 1)
        e = dgl.nn.functional.edge_softmax(graph, e)
        e = e.swapaxes(1, -1)
        graph.edata['a'] = e
        graph.ndata["ft"] = feat.unsqueeze(-1)
        graph.update_all(
            dgl.function.u_mul_e("ft", "a", "m"),
            dgl.function.sum("m", "ft"),
        )
        rst = graph.ndata["ft"]
        if self.bias is not None:
            rst = rst + self.bias.unsqueeze(-1)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst
