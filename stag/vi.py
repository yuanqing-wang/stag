import math
import torch
from .layers import StagLayer
from typing import Union

class StagMeanFieldVariationalInference(StagLayer):
    def __init__(
            self,
            p_a: torch.distributions.Distribution\
                =torch.distributions.Normal(1.0, 1.0),
            q_a_mu_init: Union[
                    float, torch.Tensor, torch.distributions.Distribution
                ]=1.0,
            q_a_log_sigma_init: Union[
                    float, torch.Tensor, torch.distributions.Distribution
                ]=math.log(1.0),
        ):
        super(StagMeanFieldVariationalInference, self).__init__()
        if isinstance(q_a_mu_init, torch.distributions.Distribution):
            q_a_mu_init = q_a_mu_init.sample()
        if isinstance(q_a_log_sigma_init, torch.distributions.Distribution):
            q_a_log_sigma_init = q_a_log_sigma_init.sample()

        self.p_a = p_a
        self.q_a_mu = torch.nn.Parameter(q_a_mu)
        self.q_a_log_sigma = torch.nn.Parameter(q_a_log_sigma)

    @property
    def q_a(self):
        return torch.distributions.Normal(
            loc=self.q_a_mu,
            scale=self.q_a_log_sigma.exp(),
        )

    def kl_divergence(self, edge_weight_sample):
        kl_divergence = self.q_a.log_prob(edge_weight_sample).sum()\
            - self.p_a.log_prob(edge_weight_sample).sum()
        return kl_divergence
