from typing import Optional, Callable
import torch
import dgl

class Stag(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        embedding_features: int=8,
        activation: Optional[Callable] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features * num_heads, bias=False)
        self.posterior_parameters = torch.nn.Linear(out_features, 4 * embedding_features)
        self.prior = torch.distributions.Normal(0.0, 1.0)
        self.embedding_features = embedding_features

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(num_heads, out_features),
            )
        else:
            self.bias = None

        self.num_heads = num_heads
        self.out_features = out_features
        self.activation = activation

    def forward(self, graph, feat):
        graph = graph.local_var()
        feat = self.fc(feat)
        feat = feat.reshape(feat.shape[:-1] + (self.num_heads, self.out_features))
        loc_l, loc_r, log_scale_l, log_scale_r = self.posterior_parameters(feat).split(self.embedding_features, dim=-1)
        graph.ndata["loc_l"] = loc_l
        graph.ndata["loc_r"] = loc_r
        graph.ndata["log_scale_l"] = log_scale_l
        graph.ndata["log_scale_r"] = log_scale_r
        graph.apply_edges(dgl.function.u_dot_v("loc_l", "loc_r", "loc"))
        graph.apply_edges(dgl.function.u_add_v("log_scale_l", "log_scale_r", "log_scale"))
        loc = graph.edata["loc"]
        scale = graph.edata["log_scale"].sum(dim=-1, keepdims=True).exp()
        e_distribution = torch.distributions.Normal(loc, scale)
        self._e_distribution = e_distribution
        e = e_distribution.rsample()
        e = dgl.nn.functional.edge_softmax(graph, e)
        graph.edata['a'] = e
        graph.ndata["ft"] = feat
        graph.update_all(
            dgl.function.u_mul_e("ft", "a", "m"),
            dgl.function.sum("m", "ft"),
        )
        rst = graph.ndata["ft"]
        if self.bias is not None:
            rst = rst + self.bias.unsqueeze(0)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst

    def kl_divergence(self):
        return torch.distributions.kl_divergence(
            self.prior,
            self._e_distribution,
        ).sum(-1).mean()

    def nll_contrastive(self, graph, feat):
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
        e_distribution = torch.distributions.Normal(loc, scale)

        fake_src = torch.randint(high=graph.number_of_nodes(), size=[graph.number_of_edges()])
        fake_dst = torch.randint(high=graph.number_of_nodes(), size=[graph.number_of_edges()])
        fake_loc = loc_l[fake_src] + loc_r[fake_dst]
        fake_scale = (log_scale_l[fake_src] + log_scale_r[fake_dst]).exp()
        fake_e_distribution = torch.distributions.Normal(fake_loc, fake_scale)
        return -e_distribution.log_prob(torch.tensor(0.0)) - fake_e_distribution.log_prob(torch.tensor(-10.0))
