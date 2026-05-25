import torch
import torch.nn.functional as F

def minimal_pad_to_divisible(tensor: torch.Tensor, sp_size: int, dim: int = 1, pad_value: float = 0.0):
    """
    Minimally pad a 3D (or higher) tensor along `dim` so its length is divisible by sp_size.

    Args:
        tensor: input PyTorch tensor (e.g. [B, L, C] or [B, H, W, C]).
        sp_size: required minimum split size.
        dim: dimension index to pad (default 1, i.e. the second dim).
        pad_value: fill value (default 0.0).

    Returns:
        (padded_tensor, padding_len): the padded tensor and the number of
        elements appended along `dim` (0 if already divisible).
    """

    current_size = tensor.size(dim)

    # Compute padding length.
    # (sp_size - current_size % sp_size) % sp_size guarantees:
    #   - 0 if current_size is already a multiple of sp_size,
    #   - otherwise the minimum number of elements needed to make it so.
    padding_len = (sp_size - current_size % sp_size) % sp_size

    if padding_len == 0:
        # Already divisible -- return as-is.
        return tensor, 0

    # Build the pad tuple.
    # torch.nn.functional.pad takes pad args from the LAST dim inward, in
    # (left, right) pairs per dim.
    # Example: tensor is [D0, D1, D2]
    #   dim=1 (D1) -> pad = (0, 0, padding_len, 0, 0, 0, ...)
    #
    # Because we pad at the end (right side) of `dim`, we need to locate
    # the right-side slot for `dim` inside the pad tuple.
    # With D = tensor.dim():
    #   dim=0 -> last two pad slots
    #   dim=1 -> pad slots index -4, -3 from the end
    #   dim=2 -> pad slots index -6, -5 from the end (for a 3D tensor, the first two)

    # 'Right pad' on the target dim. padding_dims has length 2 * D, all zeros by default.
    padding_dims = [0] * (2 * tensor.dim())

    # The 'right pad' slot for `dim` (from the END of the pad tuple, walking back).
    # Position = 2 * tensor.dim() - 2 * dim - 2
    # Example: D=3, dim=1 -> 2*3 - 2*1 - 2 = 2.
    # pad tuple is (d2_start, d2_end, d1_start, d1_end, d0_start, d0_end);
    # the value we want to fill is d1_end, at index 2.

    # F.pad expects (last_dim_start, last_dim_end, 2nd_last_start, 2nd_last_end, ...)
    # Our `dim=1` is the (D - 1 - dim) + 1 = D - dim -th dim from the end.
    # In the pad tuple it occupies indices -2*(D-dim) and -2*(D-dim)-1.
    #
    # Slot index from the LEFT (0-based):
    #   (2 * (tensor.dim() - dim - 1))     -> 'left pad'  slot
    #   (2 * (tensor.dim() - dim - 1) + 1) -> 'right pad' slot
    pad_index = 2 * (tensor.dim() - dim - 1) + 1

    if pad_index < len(padding_dims):
        padding_dims[pad_index] = padding_len
    else:
        raise ValueError("Invalid dimension index.")

    # Back to tuple.
    pad = tuple(padding_dims)

    # F.pad in 'constant' mode.
    padded_tensor = F.pad(tensor, pad=pad, mode='constant', value=pad_value)

    return padded_tensor, padding_len


def depad_by_length(padded_tensor: torch.Tensor, depadding_len: int, dim: int = 1) -> torch.Tensor:
    """
    Remove `depadding_len` trailing elements from `padded_tensor` along `dim`.

    Args:
        padded_tensor: a PyTorch tensor that has been padded.
        depadding_len: number of trailing elements to drop.
        dim: dimension index to depad (default 1, i.e. the second dim).

    Returns:
        depadded_tensor: the tensor with padding removed.
    """

    # Validate the requested length.
    current_size = padded_tensor.size(dim)
    if depadding_len < 0:
        raise ValueError("depadding_len must be non-negative.")
    if depadding_len > current_size:
        raise ValueError(f"depadding_len {depadding_len} exceeds current dim length {current_size}.")

    # Target length after removing padding.
    target_size = current_size - depadding_len

    # Build a slice tuple: full slice on every dim by default.
    slices = [slice(None)] * padded_tensor.dim()

    # On `dim`, take [0:target_size] -- i.e. drop the trailing depadding_len elements.
    slices[dim] = slice(0, target_size)

    # Tuple-unpack and slice.
    depadded_tensor = padded_tensor[tuple(slices)]

    return depadded_tensor




def pad_by_length(padded_tensor: torch.Tensor, padding_len: int, dim: int = 1, pad_value: float = 0.0) -> torch.Tensor:

    if padding_len < 0:
        raise ValueError("padding_len must be non-negative.")

    if dim < 0 or dim >= padded_tensor.dim():
        raise ValueError(f"dim index {dim} out of range [0, {padded_tensor.dim() - 1}].")

    # Build the pad tuple.
    # F.pad needs (left, right) per dim, ordered from the LAST dim inward:
    # (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right, ...).
    pad_tuple = [0] * (2 * padded_tensor.dim())

    # Right-pad on `dim` with padding_len. Translate dim index into the
    # slot inside the pad tuple (F.pad walks dims from the end).
    pad_idx = 2 * (padded_tensor.dim() - 1 - dim) + 1
    pad_tuple[pad_idx] = padding_len

    # Apply.
    padded_tensor = F.pad(padded_tensor, pad=tuple(pad_tuple), mode='constant', value=pad_value)

    return padded_tensor
