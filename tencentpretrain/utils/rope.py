import torch
from typing import Tuple

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.cpu().float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.cpu().float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).transpose(1,2).to(xq.device), xk_out.type_as(xk).transpose(1,2).to(xk.device)

def precompute_freqs_cis_new(dim: int, end: int, theta: float = 10000.0):
    # $\Thetas = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    thetas = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(end, device=thetas.device).float()

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.einsum('n,d->nd', seq_idx, thetas).float()

    # repeat so that for row $m$ we have
    # $[m \theta_0, m \theta_0, m \theta_1, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_{\frac{d}{2}}]$
    idx_theta2 = idx_theta.repeat_interleave(2, dim=-1).float()

    # Cache them
    cos_cached = idx_theta2.cos()
    sin_cached = idx_theta2.sin()

    return [cos_cached, sin_cached]


def apply_rotary_emb_new(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    cos_cached = freqs_cis[0][None, None, :, :]
    sin_cached = freqs_cis[1][None, None, :, :]

    def _neg_half(x: torch.Tensor):
        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        x0 = x[:, :, :, 0::2].reshape(-1, 1)
        xj = x[:, :, :, 1::2].reshape(-1, 1)

        return torch.stack((-xj, x0), dim = -1).reshape(x.shape)

    def _apply(x: torch.Tensor):
        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = _neg_half(x)

        return (x * cos_cached) + (neg_half_x * sin_cached)

    return _apply(xq.float()).type_as(xq), _apply(xk.float()).type_as(xk)