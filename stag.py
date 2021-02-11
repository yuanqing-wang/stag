import torch
def stag_copy_src_vi(src="h", out="m"):
    def message_fun(edges):
        if "a" in edges.data:
            return {out: edges.src[src] * edges.data["a"]}
        return {out: edges.src[src]}

    return message_fun

def stag_copy_src_normal(src='h', out='m', alpha=0.1):
    def message_fun(edges):
        h = edges.src[src]
        if h.shape[-1] >= 8:
            mask = torch.distributions.normal.Normal(
                loc=torch.tensor(1.0, device=h.device),
                scale=torch.tensor(alpha, device=h.device),
            ).sample(h.shape)
            return {out: (mask * h)}
        else:
            return {out: h}
    return message_fun


def stag_sum_dropout(msg='m', out='h', alpha=0.1):
    def reduce_func(nodes):
        m = nodes.mailbox[msg]
        if nodes._ntype == "_N":
            mask = torch.distributions.Bernoulli(
                torch.tensor(1.0-alpha, device=m.device),
            ).sample((1, 1, m.shape[-1]))

            return {out: (mask * m).sum(dim=1)}
        else:
            return {out: m.sum(dim=1)}

    return reduce_func

def stag_sum_bernoulli_shared(msg='m', out='h', alpha=0.1):
    def reduce_func(nodes):
        m = nodes.mailbox[msg]
        if nodes._ntype == "_N":
            mask = torch.distributions.Bernoulli(
                torch.tensor(1.0-alpha, device=m.device),
            ).sample((m.shape[0], m.shape[1], 1))

            mask = torch.where(
                torch.gt(mask, 0.0),
                mask / mask.sum(dim=1, keepdims=True) * mask.shape[1],
                mask
            )

            return {out: (mask * m).sum(dim=1)}
        else:
            return {out: m.sum(dim=1)}
    return reduce_func

def stag_sum_bernoulli(msg='m', out='h', alpha=0.1):
    def reduce_func(nodes):
        m = nodes.mailbox[msg]
        if nodes._ntype == "_N":
            mask = torch.distributions.Bernoulli(
                torch.tensor(1.0-alpha, device=m.device),
            ).sample(m.shape)

            mask = torch.where(
                torch.gt(mask, 0.0),
                mask / mask.sum(dim=1, keepdims=True) * mask.shape[1],
                mask
            )

            return {out: (mask * m).sum(dim=1)}
        else:
            return {out: m.sum(dim=1)}

    return reduce_func

def stag_sum_normal(msg='m', out='h', alpha=0.1):
    def reduce_func(nodes):
        m = nodes.mailbox[msg]
        mask = torch.distributions.normal.Normal(
            loc=torch.tensor(1.0, device=m.device),
            scale=torch.tensor(alpha, device=m.device),
        ).sample(m.shape)

        mask = torch.where(
            torch.gt(mask, 0.0),
            mask / mask.sum(dim=1, keepdims=True),
            mask
        )

        return {out: (mask * m).sum(dim=1)}
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

def stag_copy_src_normal_shared(src='h', out='m', alpha=0.1):
    def message_fun(edges):
        h = edges.src[src]
        if "mask" not in edges.data:
            mask = torch.distributions.normal.Normal(
                loc=torch.tensor(1.0, device=h.device),
                scale=torch.tensor(alpha, device=h.device),
            ).sample(h.shape)
            edges.data["mask"] = mask
        return {out: (edges.data["mask"] * h)}
    return message_fun


def stag_copy_src_uniform_shared(src='h', out='m', alpha=0.1):
    def message_fun(edges):
        h = edges.src[src]
        if "mask" not in edges.data:
            mask = torch.distributions.uniform.Uniform(
                high=torch.tensor(1.0+alpha, device=h.device),
                low=torch.tensor(1.0-alpha, device=h.device),
            ).sample(h.shape)
            edges.data["mask"] = mask
        return {out: (edges.data["mask"] * h)}
    return message_fun


def stag_copy_src_bernoulli_shared(src='h', out='m', alpha=0.1):
    def message_fun(edges):
        h = edges.src[src]
        if "mask" not in edges.data:
            mask = torch.distributions.bernoulli.Bernoulli(
                torch.tensor(1.0-alpha, device=h.device),
            ).sample(h.shape)
            edges.data["mask"] = mask
        return {out: (edges.data["mask"] * h)}
    return message_fun
