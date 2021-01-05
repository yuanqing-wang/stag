import torch

def stochastic_sum_normal(msg='m', out='h', alpha=0.2):
    def reducer(nodes):
        m = nodes.mailbox[msg]
        mask = torch.distributions.normal.Normal(
            loc=torch.ones_like(m),
            scale=torch.ones_like(m) * alpha,
        ).sample()
        return {out: (m * mask).sum(dim=1)}
    return reducer
