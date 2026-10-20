"""Exact mask reuse for LingBot's repeated joint VLM/action attention.

Keep the upstream FP32 Q/K/V, padding, FlexAttention kernel and output layout.
Only reuse mask construction for an unchanged tensor during one policy call.
"""
import torch
import torch.nn.functional as F


class CachedFlexAttention:
    def __init__(self):
        self.clear()

    def clear(self):
        self.source = None
        self.signature = None
        self.block_mask = None

    def __call__(self, query_states, key_states, value_states, attention_mask, scaling=None):
        from lingbotvla.models.vla.pi0.flex_attention import (
            _flex_attention, create_block_mask, create_mask, _round_up_to_multiple)
        batch, q_len, heads, dim = query_states.shape
        original_dtype = query_states.dtype
        q = query_states.transpose(1, 2).float()
        k = key_states.transpose(1, 2).float()
        v = value_states.transpose(1, 2).float()
        kv_len = k.shape[2]
        pad_q = _round_up_to_multiple(q_len, 128) - q_len
        pad_k = _round_up_to_multiple(kv_len, 128) - kv_len
        if pad_q:
            q = F.pad(q, (0, 0, 0, pad_q))
        if pad_k:
            k = F.pad(k, (0, 0, 0, pad_k))
            v = F.pad(v, (0, 0, 0, pad_k))
        signature = (attention_mask._version, tuple(attention_mask.shape),
                     batch, q_len, kv_len, heads, attention_mask.device)
        if self.source is not attention_mask or self.signature != signature:
            causal = attention_mask[:, None, :, :kv_len]
            if causal.shape[1] == 1 and heads > 1:
                causal = causal.expand(-1, heads, -1, -1)
            padded = F.pad(causal, (0, pad_k, 0, pad_q), value=0.)

            def original_mask(b, h, qi, ki):
                return padded[b][h][qi][ki]

            dense = create_mask(original_mask, B=batch, H=heads,
                                Q_LEN=q_len + pad_q, KV_LEN=kv_len + pad_k,
                                device=causal.device)

            def mask_mod(b, h, qi, ki):
                return dense[b][h][qi][ki]

            self.block_mask = create_block_mask(
                mask_mod, B=batch, H=heads, Q_LEN=q_len + pad_q, KV_LEN=kv_len + pad_k,
                BLOCK_SIZE=128, device=causal.device, _compile=False)
            self.source, self.signature = attention_mask, signature
        output = _flex_attention(
            q, k, v, block_mask=self.block_mask,
            kernel_options={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 32,
                            "BLOCK_M2": 32, "BLOCK_N2": 32, "num_stages": 2, "num_warps": 4},
            enable_gqa=True, scale=dim ** -.5 if scaling is None else scaling)
        return output[:, :, :q_len].to(original_dtype).transpose(1, 2).contiguous().reshape(batch, q_len, -1)


def check_cached_attention(device):
    from lingbotvla.models.vla.pi0.flex_attention import flex_attention_forward
    cached = CachedFlexAttention()
    results = []
    for q_len, kv_len in ((435, 435), (50, 435)):
        torch.manual_seed(28657)
        q = torch.randn(1, q_len, 16, 128, device=device, requires_grad=True)
        k = torch.randn(1, kv_len, 2, 128, device=device, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        mask = torch.ones(1, q_len, kv_len, device=device, dtype=torch.bool)
        mask[:, :, 190:210] = False
        reference = flex_attention_forward(q, k, v, mask)
        ref_grad = torch.autograd.grad(reference.square().mean(), (q, k, v))
        actual = cached(q, k, v, mask)
        got_grad = torch.autograd.grad(actual.square().mean(), (q, k, v))
        reused = cached(q, k, v, mask)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(reused, reference, rtol=1e-5, atol=1e-6)
        for got, expected in zip(got_grad, ref_grad, strict=True):
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-7)
        # Same object with changed mask values must invalidate the cache.
        previous = cached.block_mask
        mask[:, :, 10:20] = False
        changed = cached(q, k, v, mask)
        assert cached.block_mask is not previous
        torch.testing.assert_close(changed, flex_attention_forward(q, k, v, mask), rtol=1e-5, atol=1e-6)
        results.append({"q_len": q_len, "kv_len": kv_len,
                        "forward_abs": float((actual - reference).abs().max()),
                        "backward_abs": max(float((g - r).abs().max()) for g, r in zip(got_grad, ref_grad)),
                        "cache_reuse_and_mutation_checked": True})
    return results
