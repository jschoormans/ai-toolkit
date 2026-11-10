"""drop_all_true_mask must not change what the Qwen-Image transformer computes.

CPU only, tiny random-weight transformer, no model download: python testing/test_attention_mask.py
Checks (1) an all-True text mask and None give the same output and gradients, (2) a mask with padding is kept and does
change the output (so it must not be dropped).
"""
import sys

import torch
from diffusers import QwenImageTransformer2DModel
from diffusers.models.transformers import transformer_qwenimage

sys.path.insert(0, ".")
from toolkit.util.attention_mask import drop_all_true_mask


def build():
    torch.manual_seed(0)
    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        axes_dims_rope=(4, 6, 6),
    ).float()


def run(model, mask, hs, ehs, t, shapes):
    hs = hs.clone().requires_grad_(True)
    out = model(
        hidden_states=hs,
        timestep=t,
        guidance=None,
        encoder_hidden_states=ehs,
        encoder_hidden_states_mask=mask,
        img_shapes=shapes,
        return_dict=False,
    )[0]
    out.sum().backward()
    grad = hs.grad.clone()
    pgrad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    model.zero_grad(set_to_none=True)
    return out.detach(), grad, pgrad


def attn_masks_seen(model, mask, hs, ehs, t, shapes):
    """Run one forward and record whether each attention call received an attn_mask."""
    seen = []
    orig = transformer_qwenimage.dispatch_attention_fn

    def spy(q, k, v, *args, **kwargs):
        seen.append(kwargs.get("attn_mask") is not None)
        return orig(q, k, v, *args, **kwargs)

    transformer_qwenimage.dispatch_attention_fn = spy
    try:
        with torch.no_grad():
            model(hidden_states=hs, timestep=t, guidance=None, encoder_hidden_states=ehs, encoder_hidden_states_mask=mask,
                  img_shapes=shapes, return_dict=False)
    finally:
        transformer_qwenimage.dispatch_attention_fn = orig
    return seen


def main():
    model = build()
    B, T, H, W = 1, 12, 6, 6
    g = torch.Generator().manual_seed(1)
    hs = torch.randn(B, H * W, 16, generator=g)
    ehs = torch.randn(B, T, 32, generator=g)
    t = torch.tensor([0.7])
    shapes = [[(1, H, W)]] * B

    ones = torch.ones(B, T, dtype=torch.int64)
    assert drop_all_true_mask(ones) is None
    assert drop_all_true_mask(None) is None

    o_old, g_old, p_old = run(model, ones, hs, ehs, t, shapes)  # what ai-toolkit did: all-True mask
    o_new, g_new, p_new = run(model, drop_all_true_mask(ones), hs, ehs, t, shapes)  # None
    print(f"all-True vs None: output max|d| {(o_old - o_new).abs().max():.2e}, input grad max|d| {(g_old - g_new).abs().max():.2e}, "
          f"param grad max|d| {(p_old - p_new).abs().max():.2e}, bitwise equal output: {torch.equal(o_old, o_new)}")
    assert torch.allclose(o_old, o_new, atol=1e-6, rtol=1e-5)
    assert torch.allclose(g_old, g_new, atol=1e-6, rtol=1e-5)
    assert torch.allclose(p_old, p_new, atol=1e-6, rtol=1e-5)

    old_seen = attn_masks_seen(model, ones, hs, ehs, t, shapes)
    new_seen = attn_masks_seen(model, drop_all_true_mask(ones), hs, ehs, t, shapes)
    print(f"attention calls with an attn_mask: before {sum(old_seen)}/{len(old_seen)}, after {sum(new_seen)}/{len(new_seen)}")
    assert all(old_seen) and not any(new_seen)

    # batch of 2 with right padding on the second item: the mask must be kept and must matter
    B2 = 2
    hs2 = torch.randn(B2, H * W, 16, generator=g)
    ehs2 = torch.randn(B2, T, 32, generator=g)
    pad = torch.ones(B2, T, dtype=torch.int64)
    pad[1, 8:] = 0
    kept = drop_all_true_mask(pad)
    assert kept is pad
    shapes2 = [[(1, H, W)]] * B2
    t2 = torch.tensor([0.7, 0.7])
    o_mask, _, _ = run(model, kept, hs2, ehs2, t2, shapes2)
    o_none, _, _ = run(model, None, hs2, ehs2, t2, shapes2)
    diff = (o_mask[1] - o_none[1]).abs().max()
    print(f"padded item: mask kept, output differs from no-mask by {diff:.2e} (so it must not be dropped)")
    assert diff > 1e-4
    print("ok")


if __name__ == "__main__":
    main()
