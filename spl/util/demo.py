import torch
import math

##############################################################################
# 1) DCT and iDCT (Naïve Matrix Multiply Implementation)
#    - DCT-II and DCT-III with "ortho" normalization
##############################################################################

def dct_1d_naive(x: torch.Tensor, norm: str = 'ortho', dim: int = -1) -> torch.Tensor:
    """
    Naive DCT-II transform along dimension `dim`, using explicit matrix multiply.
    For smaller chunk sizes only! For large sizes, consider an FFT-based approach.
    """
    # Move `dim` to last dimension
    x = x.transpose(dim, -1)  # shape = (..., N)
    N = x.shape[-1]

    # Flatten everything except last dimension
    old_shape = x.shape[:-1]
    x_flat = x.reshape(-1, N)  # (batch, N)

    # Build the DCT-II transform matrix [N x N]:
    #   T_{k,n} = cos( π * (n+0.5) * k / N ),  k,n=0..N-1
    n = torch.arange(N, device=x.device, dtype=x.dtype).reshape(1, -1)  # (1,N)
    k = torch.arange(N, device=x.device, dtype=x.dtype).reshape(-1, 1)  # (N,1)
    cosine_mat = torch.cos((torch.pi / N) * (n + 0.5) * k)  # (N,N)

    # Matrix multiply
    X_flat = x_flat @ cosine_mat  # (batch,N)

    # Orthonormal scaling (matching scipy.fft.dct(..., type=2, norm='ortho'))
    if norm == 'ortho':
        # X[*,0] *= 1/sqrt(4N), X[*,1..] *= 1/sqrt(2N)
        X_flat[:, 0] *= 1.0 / torch.sqrt(torch.tensor(4.0*N, device=x.device, dtype=x.dtype))
        X_flat[:, 1:] *= 1.0 / torch.sqrt(torch.tensor(2.0*N, device=x.device, dtype=x.dtype))

    # Reshape and move dim back
    X = X_flat.reshape(*old_shape, N)
    X = X.transpose(dim, -1)
    return X


def idct_1d_naive(X: torch.Tensor, norm: str = 'ortho', dim: int = -1) -> torch.Tensor:
    """
    Naive Inverse DCT (DCT-III) along dimension `dim` with "ortho" normalization
    consistent with dct_1d_naive(...).
    """
    # Move `dim` to last
    X = X.transpose(dim, -1)
    N = X.shape[-1]

    # Flatten
    old_shape = X.shape[:-1]
    X_flat = X.reshape(-1, N)

    # Build iDCT (DCT-III) transform matrix:
    n = torch.arange(N, device=X.device, dtype=X.dtype).reshape(1, -1)
    k = torch.arange(N, device=X.device, dtype=X.dtype).reshape(-1, 1)
    cosine_mat = torch.cos((torch.pi / N) * (n + 0.5) * k)  # (N,N)

    if norm == 'ortho':
        # For DCT-III with orthonormal scale (matching DCT-II norm='ortho'):
        alpha_0 = 1.0 / torch.sqrt(torch.tensor(4.0*N, device=X.device, dtype=X.dtype))
        alpha_k = 1.0 / torch.sqrt(torch.tensor(2.0*N, device=X.device, dtype=X.dtype))
        alpha_vec = torch.full((N,), alpha_k, device=X.device, dtype=X.dtype)
        alpha_vec[0] = alpha_0

        # Weighted X => multiply each row by alpha(k)
        X_weighted = X_flat * alpha_vec.unsqueeze(0)
        x_flat = X_weighted @ cosine_mat
    else:
        x_flat = X_flat @ cosine_mat

    x_ = x_flat.reshape(*old_shape, N)
    x_ = x_.transpose(dim, -1)
    return x_


def dct_nd_naive(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    Applies DCT-II along *all* dims of x in succession (naïve approach).
    """
    out = x
    for d in range(x.ndim):
        out = dct_1d_naive(out, norm=norm, dim=d)
    return out


def idct_nd_naive(X: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    Applies DCT-III along *all* dims of X in succession (naïve approach).
    """
    out = X
    for d in range(X.ndim):
        out = idct_1d_naive(out, norm=norm, dim=d)
    return out


##############################################################################
# 2) Utility: Per-chunk int8 quant/dequant
##############################################################################

def float_to_int8(values: torch.Tensor):
    """
    Performs a simple per-row min-max int8 quantization on `values`, shape = (batch, k).

    Returns:
      - values_int8: (batch, k) int8
      - scales: (batch,) float
      - zero_points: (batch,) float

    The formula is:
       min_val = row.min(), max_val = row.max()
       scale = (max_val - min_val)/255
       zero_point = -min_val / scale
       q = round((val / scale) + zero_point), clamp to [-128,127]
    """
    batch, k = values.shape
    device = values.device
    dtype = values.dtype

    min_vals = values.min(dim=1).values  # (batch,)
    max_vals = values.max(dim=1).values  # (batch,)
    ranges = (max_vals - min_vals).clamp_min(1e-8)

    scales = ranges / 255.0
    zero_points = -min_vals / scales

    inv_scales = 1.0 / scales  # (batch,)
    inv_scales_2d = inv_scales.unsqueeze(-1)
    zero_points_2d = zero_points.unsqueeze(-1)

    # quantize row by row
    quant_vals_float = values * inv_scales_2d + zero_points_2d
    quant_vals_rounded = torch.round(quant_vals_float).clamp(-128, 127)
    values_int8 = quant_vals_rounded.to(torch.int8)

    return values_int8, scales, zero_points


def int8_to_float(values_int8: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor):
    """
    Dequantize row-by-row:
       val_float = (val_int8 - zero_point)*scale
    """
    batch, k = values_int8.shape
    values_int8_f = values_int8.float()
    zero_points_2d = zero_points.unsqueeze(-1)
    scales_2d = scales.unsqueeze(-1)

    out = (values_int8_f - zero_points_2d) * scales_2d
    return out


##############################################################################
# 3) Chunked DCT Encode/Decode with top-k + int8 quant
#
#    The original code assumed x.numel() is divisible by the chunk_shape product.
#    Below we patch it by adding zero-padding if needed, then storing `pad_count`.
##############################################################################

def chunked_dct_encode_int8(
    x: torch.Tensor,
    chunk_shape: tuple[int, ...],
    k: int,
    prev_error: torch.Tensor | None = None,
    norm: str = 'ortho'
):
    """
    1) If prev_error is given, add it to x.
    2) Zero-pad x if x.numel() is not divisible by product(chunk_shape).
    3) Reshape x into sub-chunks, apply nD DCT, keep top-k, int8 quantize.
    4) Return freq_idxs, freq_vals_int8, freq_scales, freq_zero_points, new_error, plus pad_count.
    """
    if prev_error is not None:
        x = x + prev_error

    full_shape = x.shape
    total_elems = x.numel()
    chunk_elems = math.prod(chunk_shape)

    # Figure out how many leftover elements we have
    remainder = total_elems % chunk_elems
    pad_count = 0
    if remainder != 0:
        pad_count = chunk_elems - remainder
        x = torch.cat([x.flatten(), x.new_zeros(pad_count)], dim=0)
    else:
        x = x.flatten()

    # shape => (num_chunks, chunk_elems)
    num_chunks = x.numel() // chunk_elems
    x_chunks = x.view(num_chunks, *chunk_shape)

    # DCT each chunk
    dct_chunks = dct_nd_naive(x_chunks, norm=norm)   # (num_chunks, *chunk_shape)
    dct_chunks_flat = dct_chunks.flatten(start_dim=1)  # (num_chunks, chunk_elems)

    # top-k
    abs_dct = dct_chunks_flat.abs()
    values, idxs = torch.topk(abs_dct, k, dim=1, largest=True, sorted=False)
    topk_vals = dct_chunks_flat.gather(dim=1, index=idxs)

    # quantize
    freq_vals_int8, scales, zero_points = float_to_int8(topk_vals)

    # reconstruct approximate DCT => measure new error
    topk_vals_approx = int8_to_float(freq_vals_int8, scales, zero_points)
    recon_flat = torch.zeros_like(dct_chunks_flat)
    recon_flat.scatter_(1, idxs, topk_vals_approx)
    recon_chunks = recon_flat.view_as(dct_chunks)
    x_recon_chunks = idct_nd_naive(recon_chunks, norm=norm)

    # flatten and remove padding
    x_recon = x_recon_chunks.view(-1)
    if pad_count > 0:
        x_recon = x_recon[:-pad_count]

    x_recon = x_recon.view(full_shape)
    # original x is also in full_shape, so compute new_error
    new_error = (x[: total_elems].view(full_shape) - x_recon)

    # Return extra pad_count so the decoder can handle it
    return idxs, freq_vals_int8, scales, zero_points, new_error, pad_count


def chunked_dct_decode_int8(
    freq_idxs: torch.Tensor,
    freq_vals_int8: torch.Tensor,
    freq_scales: torch.Tensor,
    freq_zero_points: torch.Tensor,
    x_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    norm: str = 'ortho',
    pad_count: int = 0
):
    """
    Reverse of chunked_dct_encode_int8. Accepts pad_count to remove trailing zeros.

    1) Dequant freq_vals_int8 => scatter into DCT chunks
    2) iDCT => flatten => remove padding => reshape to x_shape
    """
    total_elems = 1
    for d in x_shape:
        total_elems *= d

    chunk_elems = math.prod(chunk_shape)
    num_chunks = freq_idxs.size(0)

    freq_vals_float = int8_to_float(freq_vals_int8, freq_scales, freq_zero_points)
    recon_flat = torch.zeros(num_chunks, chunk_elems, device=freq_idxs.device, dtype=freq_vals_float.dtype)
    recon_flat.scatter_(1, freq_idxs, freq_vals_float)

    recon_chunks = recon_flat.view(num_chunks, *chunk_shape)
    x_chunks = idct_nd_naive(recon_chunks, norm=norm)
    x_approx = x_chunks.view(-1)

    if pad_count > 0:
        x_approx = x_approx[:-pad_count]

    x_approx = x_approx.view(x_shape)
    return x_approx


##############################################################################
# 4) DEMO
##############################################################################

if __name__ == "__main__":
    torch.manual_seed(0)
    shape = (256, 256)
    x = torch.randn(shape)

    chunk_shape = (64, 64)
    k = 8

    # Suppose we track error from the previous iteration
    prev_error = torch.zeros_like(x)

    # ENCODE
    (
        freq_idxs,
        freq_vals_int8,
        freq_scales,
        freq_zero_points,
        new_error,
        pad_count
    ) = chunked_dct_encode_int8(
        x, chunk_shape, k, prev_error=prev_error, norm='ortho'
    )

    # DECODE
    x_approx = chunked_dct_decode_int8(
        freq_idxs,
        freq_vals_int8,
        freq_scales,
        freq_zero_points,
        x_shape=shape,
        chunk_shape=chunk_shape,
        norm='ortho',
        pad_count=pad_count
    )

    mse = torch.mean((x - x_approx)**2).item()
    max_diff = (x - x_approx).abs().max().item()
    print(f"[INFO] MSE: {mse:.6f}")
    print(f"[INFO] Max Abs Diff: {max_diff:.6f}")
    print(f"[INFO] Residual Norm: {new_error.norm().item():.6f}")
    print(f"[INFO] pad_count used: {pad_count}")
