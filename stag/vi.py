import torch
from .layers import StagLayer

class StagVI(StagLayer):
    def __init__(
            self,
            a_prior_sigma: float=1.0,
        ):
        super(StagVI, self).__init__()
        self.a_prior_sigma = a_prior_sigma

    @property
    def p_a(self):
        return torch.distributions.Normal(1.0, self.a_prior_sigma)

    def kl_divergence(self, edge_weight_sample):
        kl_divergence = self.q_a.log_prob(edge_weight_sample).sum()\
            - self.p_a.log_prob(edge_weight_sample).sum()
        return kl_divergence
