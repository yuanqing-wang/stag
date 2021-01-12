import torch

def stag_copy_src_normal(src='h', out='m', alpha=0.1):
    if isinstance(alpha, float):
        def message_fun(edges):
            h = edges.src[src]
            mask = torch.distributions.normal.Normal(
                loc=torch.tensor(1.0, device=h.device),
                scale=torch.tensor(alpha, device=h.device),
            ).sample(h.shape)
            return {out: (mask * h)}
        return message_fun

    elif isinstance(alpha, str):
        def message_fun(edges):
            h = edges.src[src]
            if alpha in edges.dst:
                mask = torch.distributions.normal.Normal(
                    loc=torch.ones_like(edges.dst[alpha]),
                    scale=1.0-edges.dst[alpha],
                ).sample()
                return {out: (mask * h)}
            else:
                return {out: h}
        return message_fun

def stag_copy_src_uniform(src='h', out='m', alpha=0.1):
    if isinstance(alpha, float):
        def message_fun(edges):
            h = edges.src[src]
            mask = torch.distributions.uniform.Uniform(
                low=torch.tensor(1.0-alpha, device=h.device),
                high=torch.tensor(1.0+alpha, device=h.device),
            ).sample(h.shape)
            return {out: (mask * h)}
        return message_fun

def stag_copy_src_bernoulli(src='h', out='m', alpha=0.1):
    if isinstance(alpha, float):
        def message_fun(edges):
            h = edges.src[src]
            mask = torch.distributions.bernoulli.Bernoulli(
                probs=torch.tensor(1.0-alpha, device=h.device),
            ).sample(h.shape)
            return {out: (mask * h)}
        return message_fun
