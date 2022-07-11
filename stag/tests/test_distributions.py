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

def test_parametrized_distribution_expand_single():
    import torch
    import stag
    from stag.distributions import ParametrizedDistribution
    base_distribution = torch.distributions.Normal(0, 1)
    distribution = ParametrizedDistribution(base_distribution)
    sample = distribution.expand(torch.Size([10, 8])).rsample()
    assert sample.shape[0] == 10
    assert sample.shape[1] == 8

def test_parametrized_distribution_expand_multiple():
    import torch
    import stag
    from stag.distributions import ParametrizedDistribution
    base_distribution = torch.distributions.Normal(
        torch.zeros([10, 8]),
        torch.ones([10, 8])
    )
    distribution = ParametrizedDistribution(base_distribution)
    new_distribution = distribution.expand(torch.Size([12, 11, 10, 8]))
    sample = new_distribution.rsample()
    assert sample.shape[0] == 12
    assert sample.shape[1] == 11
    assert sample.shape[2] == 10
    assert sample.shape[3] == 8

def test_delta_distribution():
    import stag
    from stag.distributions import DeltaDistribution
    distribution = DeltaDistribution(0.0)
    assert distribution.sample() == 0.0
