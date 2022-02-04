import pytest

def test_import():
    import stag
    from stag import distributions
    from stag.distributions import Distribution, ParametrizedDistribution

def test_parametrized_distribution():
    import torch
    import stag
    from stag.distributions import ParametrizedDistribution
    base_distribution = torch.distributions.Normal(0, 1)
    distribution = ParametrizedDistribution(base_distribution)
    distribution_vi = ParametrizedDistribution(base_distribution, vi=True)
