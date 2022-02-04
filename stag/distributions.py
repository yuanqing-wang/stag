import torch
import math
from torch.distributions import constraints

class Distribution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def batch_shape(self):
        return self.base_distribution.batch_shape

    @property
    def mean(self):
        return self.base_distribution.mean

    @property
    def stddev(self):
        return self.base_distribution.stddev

    @property
    def variance(self):
        return self.base_distribution.variance

    def expand(self, *args, **kwargs):
        return self.base_distribution.expand(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        return self.base_distribution.rsample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.base_distribution.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.base_distribution.log_prob(*args, **kwargs)

    def cdf(self, *args, **kwargs):
        return self.base_distribution.cdf(*args, **kwargs)

    def icdf(self, *args, **kwargs):
        return self.base_distribution.icdf(*args, **kwargs)

    def entropy(self, *args, **kwargs):
        return self.base_distribution.entropy(*args, **kwargs)

    def __repr__(self):
         return repr(self.base_distribution)

class ParametrizedDistribution(Distribution):
    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
        vi: bool=False,
    ):
        super().__init__()
        parameter_names = list(base_distribution.arg_constraints.keys())

        parameters = {
            key: getattr(base_distribution, key)
            for key in parameter_names
        }

        if vi:
            new_parameter_names = []
            for key, value in parameters.items():
                if base_distribution.arg_constraints[key] == constraints.positive:
                    setattr(
                        self,
                        "log_" + key,
                        torch.nn.Parameter(torch.tensor(math.log(value))),
                    )
                    new_parameter_names.append("log_" + key)

                else:
                    setattr(self, key, torch.nn.Parameter(torch.tensor(value)))
                    new_parameter_names.append(key)

        else:
            for key, value in parameters.items():
                self.register_buffer(key, torch.tensor(value))
            new_parameter_names = parameter_names

        self.base_distribution_instance = base_distribution.__class__
        self.new_parameter_names = new_parameter_names

    @property
    def base_distribution(self):
        return self.base_distribution_instance(
            **{
                key.replace("log_", ""):getattr(self, key).exp() if "log_" in key else getattr(self, key)
                for key in self.new_parameter_names
            }
        )
