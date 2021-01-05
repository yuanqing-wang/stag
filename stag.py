import torch

def stochastic_sum_normal(msg='m', out='h', alpha=0.2):
    def reducer(nodes):
        m = nodes.mailbox[msg]
        mask = torch.distributions.normal.Normal(
            loc=torch.tensor(1.0, device=m.device),
            scale=torch.tensor(alpha, device=m.device),
        ).sample(m.shape)
        return {out: (m * mask).sum(dim=1)}
    return reducer

def stochastic_sum_uniform(msg='m', out='h', alpha=0.2):
    def reducer(nodes):
        m = nodes.mailbox[msg]
        mask = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0-alpha, device=m.device),
            high=torch.tensor(1.0+alpha, device=m.device),
        ).sample(m.shape)
        return {out: (m * mask).sum(dim=1)}
    return reducer
