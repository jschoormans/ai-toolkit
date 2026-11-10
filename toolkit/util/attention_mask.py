from typing import Optional

import torch


def drop_all_true_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return None when a text padding mask marks every token as valid, otherwise return the mask unchanged.

    An all-True mask means "attend to everything", the same as no mask. Passing None lets the diffusers Qwen-Image
    transformer skip building a joint attention mask, so scaled_dot_product_attention can use the flash kernel
    (it falls back to a slower kernel whenever an attn_mask is given). With batch size 1 there is no padding, so the
    mask is always all True. With padding the mask is returned as is.

    The check is one tiny reduction per forward pass.
    """
    if mask is None:
        return None
    if bool(mask.bool().all()):
        return None
    return mask
