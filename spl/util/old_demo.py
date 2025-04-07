import torch
import math
from ..device import device
import time

##############################################################################
# 0) Helper: Create the DCT Matrix (with “ortho” scaling)
##############################################################################

def _create_dct_matrix(N: int, norm: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build the DCT-II matrix of shape (N, N). If norm=='ortho' then the matrix
    is scaled such that the transform is orthonormal.
    """
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # shape (1, N)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # shape (N, 1)
    # Compute the cosine basis: cos(π*(n+0.5)*k/N)
    D = torch.cos(math.pi * (n + 0.5) * k / N)  # shape (N, N)
    if norm == 'ortho':
        # For k=0, factor becomes 1/sqrt(2)
        D[0, :] = D[0, :] / math.sqrt(2)
        # Scale entire matrix to achieve orthonormality
        D = D * math.sqrt(2.0 / N)
    return D

##############################################################################
# 1) DCT and iDCT (Matrix Multiply Implementation with Correct “ortho” scaling)
##############################################################################

def dct_1d_naive(x: torch.Tensor, norm: str = 'ortho', dim: int = -1) -> torch.Tensor:
    """
    Compute the 1D DCT-II of x along dimension `dim` using an explicit matrix
    multiplication. For an input vector x of length N, we define:
    
       X[k] = sqrt((2 - δ[k,0]) / N) * Σ x[n] cos(π*(n+0.5)*k/N)
       
    which is orthonormal when norm=='ortho'.
    """
    # Bring the desired dimension to the end.
    x = x.transpose(dim, -1)
    N = x.shape[-1]
    old_shape = x.shape[:-1]
    x_flat = x.reshape(-1, N)
    device, dtype = x.device, x.dtype

    # Compute the DCT-II matrix.
    D = _create_dct_matrix(N, norm, device, dtype)
    # Multiply by the transpose of the DCT matrix.
    X_flat = x_flat @ D.t()
    X = X_flat.view(*old_shape, N).transpose(dim, -1)
    return X


def idct_1d_naive(X: torch.Tensor, norm: str = 'ortho', dim: int = -1) -> torch.Tensor:
    """
    Compute the 1D inverse DCT (DCT-III) along dimension `dim`.
    With the above orthonormal definition the inverse is given by:
    
       x = X @ D,
       
    where D is the same DCT matrix used in dct_1d_naive.
    """
    X = X.transpose(dim, -1)
    N = X.shape[-1]
    old_shape = X.shape[:-1]
    X_flat = X.reshape(-1, N)
    device, dtype = X.device, X.dtype

    # Build the same DCT matrix.
    D = _create_dct_matrix(N, norm, device, dtype)
    # Inverse DCT via matrix multiplication.
    x_flat = X_flat @ D
    x = x_flat.view(*old_shape, N).transpose(dim, -1)
    return x


def dct_nd_naive(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    Apply the 1D DCT-II along every dimension in succession.
    """
    out = x
    for d in range(x.ndim):
        out = dct_1d_naive(out, norm=norm, dim=d)
    return out


def idct_nd_naive(X: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    Apply the 1D inverse DCT (DCT-III) along every dimension in succession.
    """
    out = X
    for d in range(X.ndim):
        out = idct_1d_naive(out, norm=norm, dim=d)
    return out


##############################################################################
# 2) Utility: Per-chunk int8 Quantization/Dequantization (Corrected Mapping)
##############################################################################

def float_to_int8(values: torch.Tensor):
    """
    For each row (of shape (batch, k)) of `values`, quantize linearly from the
    range [min, max] to the int8 range [-128, 127].
    
    The mapping is:
         scale = (max - min) / 255,
         q = round((value - min) / scale) - 128,
         clamped to [-128, 127].
    
    Returns:
      - values_int8: (batch, k) int8 tensor,
      - scales: (batch,) tensor,
      - min_vals: (batch,) tensor.
    """
    batch, k = values.shape
    # Keep dims for broadcasting.
    min_vals = values.min(dim=1, keepdim=True).values  # shape (batch, 1)
    max_vals = values.max(dim=1, keepdim=True).values  # shape (batch, 1)
    ranges = (max_vals - min_vals).clamp_min(1e-8)
    scales = ranges / 255.0  # shape (batch, 1)
    
    # Map values so that: value = min -> q = -128, value = max -> q = 127.
    q = torch.round((values - min_vals) / scales) - 128
    q = q.clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(-1), min_vals.squeeze(-1)


def int8_to_float(q: torch.Tensor, scales: torch.Tensor, min_vals: torch.Tensor):
    """
    Dequantize each row using the inverse mapping:
         value = (q + 128) * scale + min.
    """
    q_f = q.float()
    return (q_f + 128) * scales.unsqueeze(-1) + min_vals.unsqueeze(-1)


##############################################################################
# 3) Chunked DCT Encode/Decode with Top-k + int8 Quantization
##############################################################################

def chunked_dct_encode_int8(
    x: torch.Tensor,
    chunk_shape: torch.Tensor,
    k: torch.Tensor,
    prev_error: torch.Tensor | None = None,
    norm: str = 'ortho'
):
    """
    1) Optionally add prev_error to x.
    2) Zero-pad x so that the total number of elements is divisible by the product of chunk_shape.
    3) Reshape x into chunks, apply the nD DCT, select the top-k coefficients (by absolute value)
       in each chunk, and quantize them to int8.
    4) Reconstruct an approximate DCT (by dequantizing and scattering the values back)
       to compute the residual error.
    5) Returns freq_idxs, freq_vals_int8, scales, min_vals, new_error, and pad_count.

    Note:
      - chunk_shape is a 1D torch.Tensor containing the dimensions of each chunk.
      - k is a scalar torch.Tensor representing the number of coefficients to retain.
    """
    if prev_error is not None:
        x = x + prev_error

    full_shape = x.shape
    total_elems = x.numel()
    chunk_elems = int(torch.prod(chunk_shape).item())
    
    # Save the original flattened x for residual error computation.
    x_orig_flat = x.flatten()

    # Zero-pad if necessary.
    remainder = total_elems % chunk_elems
    pad_count = 0
    if remainder != 0:
        pad_count = chunk_elems - remainder
        x = torch.cat([x_orig_flat, x.new_zeros(pad_count)], dim=0)
    else:
        x = x_orig_flat

    num_chunks = x.numel() // chunk_elems
    # Convert chunk_shape to a list of ints
    chunk_shape_int = [int(s) for s in chunk_shape.tolist()]
    x_chunks = x.view(num_chunks, *chunk_shape_int)

    # Apply the nD DCT on each chunk.
    dct_chunks = dct_nd_naive(x_chunks, norm=norm)   # shape: (num_chunks, *chunk_shape)
    dct_chunks_flat = dct_chunks.flatten(start_dim=1)  # shape: (num_chunks, chunk_elems)

    # Select top-k coefficients (by absolute value) in each chunk.
    abs_dct = dct_chunks_flat.abs()
    k_val = int(k.item())
    _, freq_idxs = torch.topk(abs_dct, k_val, dim=1, largest=True, sorted=False)
    topk_vals = dct_chunks_flat.gather(dim=1, index=freq_idxs)

    # Quantize the top-k coefficients.
    freq_vals_int8, scales, min_vals = float_to_int8(topk_vals)

    # Reconstruct approximate DCT from the quantized values.
    topk_vals_approx = int8_to_float(freq_vals_int8, scales, min_vals)
    recon_flat = torch.zeros_like(dct_chunks_flat)
    recon_flat.scatter_(1, freq_idxs, topk_vals_approx)
    recon_chunks = recon_flat.view_as(dct_chunks)

    # Invert the DCT to obtain the approximate x.
    x_recon_chunks = idct_nd_naive(recon_chunks, norm=norm)
    x_recon_flat = x_recon_chunks.reshape(-1)
    if pad_count > 0:
        x_recon_flat = x_recon_flat[:-pad_count]
    x_recon = x_recon_flat.view(full_shape)

    # Compute the residual error (using the original x without padding).
    new_error = x_orig_flat[:total_elems].view(full_shape) - x_recon
    return freq_idxs, freq_vals_int8, scales, min_vals, new_error, torch.tensor(pad_count, device=x.device)


def chunked_dct_decode_int8(
    freq_idxs: torch.Tensor,
    freq_vals_int8: torch.Tensor,
    scales: torch.Tensor,
    min_vals: torch.Tensor,
    x_shape: tuple[int, ...],
    chunk_shape,
    norm: str,
    pad_count: torch.Tensor
):
    """
    Decode the chunked DCT:
      1) Dequantize freq_vals_int8.
      2) Scatter them back into the DCT coefficient array.
      3) Apply the inverse nD DCT.
      4) Remove zero-padding and reshape to x_shape.

    Note:
      - chunk_shape can be either a 1D torch.Tensor or a tuple containing the dimensions of each chunk.
    """
    # Ensure chunk_shape is a tensor
    if not isinstance(chunk_shape, torch.Tensor):
        chunk_shape = torch.tensor(chunk_shape, dtype=torch.int64, device=freq_idxs.device)
    chunk_elems = int(torch.prod(chunk_shape).item())
    num_chunks = freq_idxs.size(0)

    freq_vals_float = int8_to_float(freq_vals_int8, scales, min_vals)
    recon_flat = torch.zeros(num_chunks, chunk_elems, device=freq_idxs.device,
                               dtype=freq_vals_float.dtype)
    recon_flat.scatter_(1, freq_idxs, freq_vals_float)
    # Convert chunk_shape to a list of ints
    chunk_shape_int = [int(s) for s in chunk_shape.tolist()]
    recon_chunks = recon_flat.view(num_chunks, *chunk_shape_int)
    x_chunks = idct_nd_naive(recon_chunks, norm=norm)
    x_approx_flat = x_chunks.reshape(-1)
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

##############################################################################
# 4) DEMO
##############################################################################

if __name__ == "__main__":
    torch.manual_seed(0)
    shape = (6000, 6000)
    x = torch.randn(shape)

    # Use chunks of shape 64x64 (each chunk has 4096 elements).
    # With k=4096 you retain all coefficients.
    chunk_shape = torch.tensor([64, 64])
    k = torch.tensor(4096)

    # (Optional) Previous error – here zero.
    prev_error = torch.zeros_like(x)
    
    
    start = time.time()

    # ENCODE
    (freq_idxs,
     freq_vals_int8,
     scales,
     min_vals,
     new_error,
     pad_count) = chunked_dct_encode_int8(
                        x, chunk_shape, k, prev_error=prev_error, norm='ortho')
     
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
                   pad_count=pad_count)

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
