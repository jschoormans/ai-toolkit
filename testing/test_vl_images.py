"""The VL condition image helper must give the Qwen2-VL processor the same pixels as a plain PIL image.

Runs on CPU without any model weights: PYTHONPATH=. python testing/test_vl_images.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen2VLImageProcessor

from toolkit.util.vl_images import tensor_to_vl_pil, vl_condition_size


def make_processor():
    return Qwen2VLImageProcessor(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )


def test_helper_matches_pil():
    proc = make_processor()
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, (448, 640, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    t = torch.from_numpy(arr).permute(2, 0, 1).float()[None] / 255  # 0-1, as ai-toolkit loads control images

    reference = proc(images=[pil.resize((640, 448), Image.LANCZOS)], return_tensors="pt")["pixel_values"]
    fixed = proc(images=[tensor_to_vl_pil(t, 448, 640)], return_tensors="pt")["pixel_values"]
    raw_tensor = proc(images=[t], return_tensors="pt")["pixel_values"]  # what the old code path did

    assert torch.allclose(fixed, reference), "helper output differs from the plain PIL path"
    assert (raw_tensor - reference).abs().max() > 1.0, "expected the raw 0-1 tensor path to be wrong"
    print(f"ok: fixed vs reference max|d| {(fixed - reference).abs().max():.1e}, "
          f"raw tensor mean {raw_tensor.mean():.2f} vs reference {reference.mean():.2f}")


def test_condition_size_512():
    """model_kwargs.condition_image_size 262144 (512x512): the training path must hand the processor
    the same pixels as a plain PIL image resized to the condition size; the old tensor path must not."""
    proc = make_processor()
    rng = np.random.RandomState(1)
    for h, w in [(1024, 1024), (1024, 768)]:
        arr = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        pil = Image.fromarray(arr)
        t = torch.from_numpy(arr).permute(2, 0, 1).float()[None] / 255

        ch, cw = vl_condition_size(h, w, 262144)
        if h == w:
            assert (ch, cw) == (512, 512), (ch, cw)
        reference = proc(images=[pil.resize((cw, ch), Image.LANCZOS)], return_tensors="pt")
        fixed = proc(images=[tensor_to_vl_pil(t, ch, cw)], return_tensors="pt")
        old = proc(  # pre-fix code path: bilinear 0-1 tensor straight into the processor
            images=[F.interpolate(t, size=(ch, cw), mode="bilinear")], return_tensors="pt"
        )
        default = proc(images=[pil.resize(vl_condition_size(h, w, 384 * 384)[::-1], Image.LANCZOS)], return_tensors="pt")

        assert torch.equal(fixed["pixel_values"], reference["pixel_values"]), "fixed path differs from PIL"
        assert torch.equal(fixed["image_grid_thw"], reference["image_grid_thw"])
        assert (old["pixel_values"] - reference["pixel_values"]).abs().max() > 1.0, "old tensor path should differ"
        assert fixed["pixel_values"].shape[0] > default["pixel_values"].shape[0], "512 should give more patches than 384"
        print(f"ok: control {h}x{w} -> VL image {ch}x{cw}, grid_thw {fixed['image_grid_thw'].tolist()} "
              f"pixel_values {tuple(fixed['pixel_values'].shape)} (384^2 default: {tuple(default['pixel_values'].shape)}); "
              f"fixed vs PIL max|d| {(fixed['pixel_values'] - reference['pixel_values']).abs().max():.1e}; "
              f"old tensor path mean {old['pixel_values'].mean():.2f} vs PIL {reference['pixel_values'].mean():.2f}")


if __name__ == "__main__":
    test_helper_matches_pil()
    test_condition_size_512()
