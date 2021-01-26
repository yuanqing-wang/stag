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
                    scale=edges.dst[alpha],
                ).sample()
                return {out: (mask * h)}
            else:
                return {out: h}
        return message_fun


def stag_sum_normal(msg='m', out='h', alpha=0.1):
    if isinstance(alpha, float):
        def reduce_func(nodes):
            m = nodes.mailbox[msg]
            mask = torch.distributions.normal.Normal(
                loc=torch.tensor(1.0, device=m.device),
                scale=torch.tensor(alpha, device=m.device),
            ).sample(m.shape)
            mask /= mask.sum(dim=1, keepdims=True)            

            return {out: (mask * m).sum(dim=1)}
        return reduce_func

    elif isinstance(alpha, str):
        def reduce_func(nodes):
            m = nodes.mailbox[msg]
            if alpha in nodes.data:
                mask = torch.distributions.normal.Normal(
                   loc=torch.ones_like(m),
                   scale=nodes.data[alpha][:, None, :],
                ).sample()
                mask /= mask.sum(dim=1, keepdims=True)

                return {out: (mask * m).sum(dim=1)}
            else:
                return {out: m.sum(dim=1)}
        return reduce_func

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
