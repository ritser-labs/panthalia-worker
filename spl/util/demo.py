import torch
import math
import time
from ..device import device

################################################################################
# 0) Caching the 2D DCT Matrix
################################################################################

# Global cache: {(C, norm, device, dtype): (D_matrix, D_t_matrix)}
_2D_DCT_CACHE = {}

def _create_2d_dct_mats(C: int, norm: str, device: torch.device, dtype: torch.dtype):
    """
    Build (or retrieve from cache) the 2D DCT factor:
       D of shape (C, C)
    with the standard 'naive' definition (half-sample shift).
    For 'ortho' scaling, we do the usual sqrt(2/C) and 1/sqrt(2) for the first row.
    """
    key = (C, norm, device, dtype)
    if key in _2D_DCT_CACHE:
        return _2D_DCT_CACHE[key]

    # 1D DCT of size (C, C)
    n = torch.arange(C, device=device, dtype=dtype).unsqueeze(0)  # (1, C)
    k = torch.arange(C, device=device, dtype=dtype).unsqueeze(1)  # (C, 1)
    D1d = torch.cos(math.pi*(n + 0.5)*k / C)  # shape (C, C)
    if norm == 'ortho':
        # for k=0 row => / sqrt(2)
        D1d[0, :] *= 1.0 / math.sqrt(2.0)
        # then multiply entire matrix by sqrt(2.0/C)
        D1d *= math.sqrt(2.0/C)

    # We'll store both D and D^T (D_t)
    D_t = D1d.t().contiguous()

    _2D_DCT_CACHE[key] = (D1d, D_t)
    return D1d, D_t

################################################################################
# 1) Batched 2D DCT / iDCT
################################################################################

def dct_2d_batched(x: torch.Tensor, D: torch.Tensor, D_t: torch.Tensor) -> torch.Tensor:
    """
    2D DCT for a batch of chunks x, shape (B, C, C).

    We do: X = D * x * D^T in batched form:
      1) X_mid = D_expand bmm x
      2) X_out = X_mid bmm D_expand_t
    which yields shape (B, C, C).
    """
    B, C, C2 = x.shape
    assert C == C2, "Chunks must be (C, C)."

    # Expand D to (B, C, C) but broadcast the same D for all B
    D_expand = D.unsqueeze(0).expand(B, -1, -1)        # (B, C, C)
    D_expand_t = D_t.unsqueeze(0).expand(B, -1, -1)    # (B, C, C)

    # X_mid = D x
    X_mid = torch.bmm(D_expand, x)         # (B, C, C)
    # X_out = X_mid D^T
    X_out = torch.bmm(X_mid, D_expand_t)   # (B, C, C)
    return X_out

def idct_2d_batched(X: torch.Tensor, D: torch.Tensor, D_t: torch.Tensor) -> torch.Tensor:
    """
    2D iDCT for a batch of chunks X, shape (B, C, C):
       x = D^T * X * D
    in batched form:
      1) X_mid = X bmm D
      2) x = D^T bmm X_mid
    """
    B, C, C2 = X.shape
    assert C == C2, "Chunks must be (C, C)."

    D_expand = D.unsqueeze(0).expand(B, -1, -1)        # (B, C, C)
    D_expand_t = D_t.unsqueeze(0).expand(B, -1, -1)    # (B, C, C)

    # X_mid = X * D
    X_mid = torch.bmm(X, D_expand)         # (B, C, C)
    # x_approx = D^T * X_mid
    x_approx = torch.bmm(D_expand_t, X_mid) # (B, C, C)
    return x_approx

################################################################################
# 2) Per-chunk int8 Quantization
################################################################################

def float_to_int8(values: torch.Tensor):
    batch, k = values.shape
    min_vals = values.min(dim=1, keepdim=True).values
    max_vals = values.max(dim=1, keepdim=True).values
    ranges = (max_vals - min_vals).clamp_min(1e-8)
    scales = ranges / 255.0

    q = torch.round((values - min_vals) / scales) - 128
    q = q.clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(-1), min_vals.squeeze(-1)

def int8_to_float(q: torch.Tensor, scales: torch.Tensor, min_vals: torch.Tensor):
    return (q.float() + 128)*scales.unsqueeze(-1) + min_vals.unsqueeze(-1)

################################################################################
# 3) Chunked 2D DCT Encode/Decode
################################################################################

def chunked_dct_encode_int8(
    x: torch.Tensor,
    chunk_shape: torch.Tensor,
    k: torch.Tensor,
    prev_error: torch.Tensor|None=None,
    norm: str='ortho'
):
    """
    Faster 2D DCT-based encode for chunk_shape=(C, C):
      1) Optionally add prev_error
      2) Zero-pad if needed
      3) Reshape into (num_chunks, C, C)
      4) 2D DCT in a single batched matmul: X = D * chunk * D^T
      5) top-k, int8 quantize
      6) Reconstruct approximate freq => iDCT => compute new_error
    """
    if prev_error is not None:
        x = x + prev_error

    full_shape = x.shape
    total_elems = x.numel()
    chunk_elems = int(torch.prod(chunk_shape).item())  # C*C

    C1, C2 = chunk_shape.tolist()
    assert C1 == C2, "This code only supports chunk_shape=(C, C)."
    C = C1

    x_orig_flat = x.flatten()

    remainder = total_elems % chunk_elems
    pad_count = 0
    if remainder != 0:
        pad_count = chunk_elems - remainder
        x_padded = torch.cat([x_orig_flat, x.new_zeros(pad_count)], dim=0)
    else:
        x_padded = x_orig_flat

    num_chunks = x_padded.numel() // chunk_elems
    x_chunks_2d = x_padded.view(num_chunks, C, C)  # shape (B, C, C)

    # Build / retrieve the 2D DCT matrix
    D, D_t = _create_2d_dct_mats(C, norm, x.device, x.dtype)

    # 2D DCT in a single batched matmul
    dct_chunks_2d = dct_2d_batched(x_chunks_2d, D, D_t)  # shape (B, C, C)

    # Flatten each chunk => shape (B, C*C)
    dct_chunks_flat = dct_chunks_2d.reshape(num_chunks, -1)
    abs_dct = dct_chunks_flat.abs()

    k_val = int(k.item())
    _, freq_idxs = torch.topk(abs_dct, k_val, dim=1, largest=True, sorted=False)
    topk_vals = dct_chunks_flat.gather(dim=1, index=freq_idxs)

    freq_vals_int8, scales, min_vals = float_to_int8(topk_vals)

    # Reconstruct approximate freq
    topk_vals_approx = int8_to_float(freq_vals_int8, scales, min_vals)

    recon_flat = torch.zeros_like(dct_chunks_flat)
    recon_flat.scatter_(1, freq_idxs, topk_vals_approx)

    # iDCT: x_approx = D^T * freq * D
    recon_chunks_2d = recon_flat.view(num_chunks, C, C)
    x_recon_2d = idct_2d_batched(recon_chunks_2d, D, D_t)  # NOTE: pass (D, D_t) in this order
    x_recon_flat = x_recon_2d.reshape(-1)

    if pad_count > 0:
        x_recon_flat = x_recon_flat[:-pad_count]
    x_recon = x_recon_flat.view(full_shape)

    # Residual
    new_error = x_orig_flat[:total_elems].view(full_shape) - x_recon

    return freq_idxs, freq_vals_int8, scales, min_vals, new_error, torch.tensor(pad_count, device=x.device)


def chunked_dct_decode_int8(
    freq_idxs: torch.Tensor,
    freq_vals_int8: torch.Tensor,
    scales: torch.Tensor,
    min_vals: torch.Tensor,
    x_shape: tuple[int,...],
    chunk_shape,
    norm: str,
    pad_count: torch.Tensor
):
    """
    2D decode from chunked DCT:
      1) freq_vals_int8 -> float
      2) scatter => freq
      3) iDCT in a single batched matmul: x = D^T * freq * D
      4) remove padding
    """
    if not isinstance(chunk_shape, torch.Tensor):
        chunk_shape = torch.tensor(chunk_shape, dtype=torch.int64, device=freq_idxs.device)
    C1, C2 = chunk_shape.tolist()
    assert C1 == C2, "This code only supports chunk_shape=(C, C)."
    C = C1

    chunk_elems = C*C
    num_chunks = freq_idxs.size(0)

    freq_vals_float = int8_to_float(freq_vals_int8, scales, min_vals)
    recon_flat = torch.zeros(num_chunks, chunk_elems, device=freq_idxs.device, dtype=freq_vals_float.dtype)
    recon_flat.scatter_(1, freq_idxs, freq_vals_float)

    recon_chunks_2d = recon_flat.view(num_chunks, C, C)

    # Build / retrieve the DCT matrix
    D, D_t = _create_2d_dct_mats(C, norm, freq_idxs.device, freq_idxs.dtype)

    # iDCT: x = D^T * freq * D
    x_chunks_2d = idct_2d_batched(recon_chunks_2d, D, D_t)
    x_approx_flat = x_chunks_2d.reshape(-1)

    pad_count_val = pad_count.item()
    if pad_count_val > 0:
        x_approx_flat = x_approx_flat[:-pad_count_val]
    x_approx = x_approx_flat.view(x_shape)
    return x_approx

async def decode_diff(diff_data: dict) -> torch.Tensor:
    freq_idxs_t = diff_data['freq_idxs'].clone().detach().to(device=device, dtype=torch.int64)
    freq_vals_int8_t = diff_data['freq_vals_int8'].clone().detach().to(device=device, dtype=torch.int8)
    freq_scales_t = diff_data['freq_scales'].clone().detach().to(device=device, dtype=torch.float32)
    freq_zero_points_t = diff_data['freq_zero_points'].clone().detach().to(device=device, dtype=torch.float32)

    chunk_shape = tuple(diff_data['chunk_shape'])
    orig_shape = tuple(diff_data['orig_shape'].tolist())
    pad_count = diff_data.get('pad_count', 0)

    param_diff = chunked_dct_decode_int8(
        freq_idxs_t,
        freq_vals_int8_t,
        freq_scales_t,
        freq_zero_points_t,
        x_shape=orig_shape,
        chunk_shape=chunk_shape,
        norm='ortho',
        pad_count=pad_count
    )

    param_diff = param_diff.to(device=device, dtype=torch.float32)
    return param_diff

################################################################################
# 4) Demo
################################################################################

if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (6000, 6000)
    x = torch.randn(shape)

    # chunk_shape=(64,64) => 4096 elems
    chunk_shape = torch.tensor([64, 64])
    # with k=4096 => no freq is dropped
    k = torch.tensor(4096)
    prev_error = torch.zeros_like(x)

    start = time.time()

    # ENCODE
    (freq_idxs,
     freq_vals_int8,
     scales,
     min_vals,
     new_error,
     pad_count) = chunked_dct_encode_int8(
         x, chunk_shape, k, prev_error=prev_error, norm='ortho'
    )

    encode_end = time.time()
    print(f"[INFO] Encoding time: {encode_end - start:.6f} seconds")

    # DECODE
    x_approx = chunked_dct_decode_int8(
        freq_idxs,
        freq_vals_int8,
        scales,
        min_vals,
        x_shape=shape,
        chunk_shape=chunk_shape,
        norm='ortho',
        pad_count=pad_count
    )

    decode_end = time.time()
    print(f"[INFO] Decoding time: {decode_end - encode_end:.6f} seconds")
    print(f"[INFO] Total time:    {decode_end - start:.6f} seconds")

    mse = torch.mean((x - x_approx)**2).item()
    max_diff = (x - x_approx).abs().max().item()
    residual_norm = new_error.norm().item()

    print(f"[INFO] x values:       {x.flatten()[:10]}")
    print(f"[INFO] x_approx vals:  {x_approx.flatten()[:10]}")
    print(f"[INFO] MSE:            {mse:.6f}")
    print(f"[INFO] Max Abs Diff:   {max_diff:.6f}")
    print(f"[INFO] Residual Norm:  {residual_norm:.6f}")
    print(f"[INFO] pad_count used: {pad_count.item()}")
