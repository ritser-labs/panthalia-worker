#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

# If needed, install: pip install torch-dct
import torch_dct as dct


###############################################################################
# GLOBALS for measuring bandwidth
###############################################################################
cluster_bandwidth = 0.0  # Accumulates total communication cost (in bytes), rank=0 owns it.

def measure_bytes(tensor: torch.Tensor) -> int:
    """Returns the number of bytes used by 'tensor'."""
    return tensor.numel() * tensor.element_size()

def measure_all_gather(tensor: torch.Tensor, world_size: int) -> int:
    """
    In an all_gather, each rank sends a copy of 'tensor' to every other rank,
    and each rank receives everyone else's data.
    Approx cluster-wide cost = world_size^2 * measure_bytes(tensor).
    """
    return world_size * world_size * measure_bytes(tensor)

def measure_broadcast(tensor: torch.Tensor, world_size: int) -> int:
    """
    Broadcasting from rank=0 to others has total cost ~ (world_size-1)*measure_bytes(tensor).
    """
    return (world_size - 1) * measure_bytes(tensor)

def record_bandwidth(bytes_to_add: int, rank: int):
    """
    Only rank=0 accumulates it, for the entire cluster.
    """
    global cluster_bandwidth
    if rank == 0:
        cluster_bandwidth += bytes_to_add


###############################################################################
# DISTRIBUTED SETUP
###############################################################################
def setup_distributed(rank, world_size, backend="gloo"):
    dist_timeout = datetime.timedelta(minutes=5)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, timeout=dist_timeout
    )
    print(f"[Rank {rank}] init with backend={backend}, world_size={world_size}", flush=True)

def cleanup_distributed():
    dist.destroy_process_group()


###############################################################################
# A TINY MODEL
###############################################################################
class TinyLM(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: (B, L) of token IDs
        returns logits of shape (B*L, vocab_size)
        """
        emb = self.emb(x)             # (B, L, hidden_dim)
        out = self.fc1(emb)           # (B, L, hidden_dim)
        out = self.act(out)           # (B, L, hidden_dim)
        out = self.fc2(out)           # (B, L, vocab_size)
        out = out.view(-1, out.size(-1))
        return out


###############################################################################
# DATA (Now with a strong "go up" pattern)
###############################################################################
def generate_synthetic_data(num_sequences=2000, seq_len=20, vocab_size=100, seed=42):
    """
    Creates a toy dataset where each sequence "goes up" in a cycle:
     seq[i] = (offset + i) mod vocab_size

    offset is randomly chosen per sequence in [0..vocab_size-1].
    This ensures a consistent next-token relationship.
    """
    random.seed(seed)
    data = []
    for _ in range(num_sequences):
        offset = random.randint(0, vocab_size - 1)
        seq = []
        for i in range(seq_len):
            val = (offset + i) % vocab_size
            seq.append(val)
        data.append(torch.tensor(seq, dtype=torch.long))
    return data

def infinite_data_loader(data, batch_size=8, shuffle=True):
    idx = 0
    order = list(range(len(data)))
    if shuffle:
        random.shuffle(order)
    while True:
        if idx >= len(data):
            idx = 0
            if shuffle:
                random.shuffle(order)
        seq = data[order[idx]]
        idx += 1
        yield pad_sequence([seq], batch_first=True, padding_value=0)

def simple_data_loader(data, batch_size=8, shuffle=False):
    order = list(range(len(data)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        batch_idxs = order[i:i+batch_size]
        batch_seqs = [data[idx] for idx in batch_idxs]
        yield pad_sequence(batch_seqs, batch_first=True, padding_value=0)


###############################################################################
# LOSS (Next-token prediction)
###############################################################################
def lm_loss_fn(logits, x):
    """
    Each position tries to predict x[i+1].
    The last token in each sequence is ignored (=-100).
    """
    B, L = x.shape
    target = x.clone()
    for i in range(B):
        for j in range(L - 1):
            target[i, j] = x[i, j+1]
        target[i, L-1] = -100  # ignore
    target = target.view(-1)
    return nn.CrossEntropyLoss(ignore_index=-100)(logits, target)


###############################################################################
# INT8 QUANT/DEQUANT
###############################################################################
def quantize_int8(tensor_cpu):
    min_val = tensor_cpu.min()
    max_val = tensor_cpu.max()
    range_val = max_val - min_val
    if range_val < 1e-8:
        range_val = 1e-8
    scale = range_val / 255.0
    zero_point = min_val
    q_fp = (tensor_cpu - zero_point) / scale
    q_rounded = torch.clamp(torch.round(q_fp - 128), -128, 127)
    q_int8 = q_rounded.to(torch.int8)
    return q_int8, scale, zero_point

def dequantize_int8(q_int8_cpu, scale, zero_point):
    q_fp = q_int8_cpu.float() + 128.0
    return q_fp * scale + zero_point


###############################################################################
# DCT VIA torch_dct
###############################################################################
def dct_1d_torch_dct(x_cpu):
    return dct.dct(x_cpu, norm='ortho')

def idct_1d_torch_dct(X_cpu):
    return dct.idct(X_cpu, norm='ortho')


###############################################################################
# DE-MO FAST COMPONENTS
###############################################################################
def extract_fast_components_cpu(delta_cpu, k_top):
    """
    1) Do a DCT on 'delta_cpu'
    2) Keep only top-k magnitude freq components
    """
    print(f"[Rank {dist.get_rank()}] about to do torch_dct.dct() for param size={delta_cpu.numel()}", flush=True)
    dct_vals = dct_1d_torch_dct(delta_cpu)
    mag = dct_vals.abs()
    total_sz = dct_vals.numel()
    if k_top > total_sz:
        k_top = total_sz
    topk = torch.topk(mag, k_top, largest=True)
    indices = topk.indices
    amps = dct_vals[indices]
    return indices, amps

def reconstruct_fast_components_cpu(indices_cpu, amps_cpu, total_size):
    """
    Reconstruct a vector of length total_size with 'amps_cpu' placed
    at 'indices_cpu', then do an inverse DCT.
    """
    dct_vals = torch.zeros(total_size, dtype=amps_cpu.dtype)
    dct_vals[indices_cpu] = amps_cpu
    print(f"[Rank {dist.get_rank()}] about to do torch_dct.idct() for param size={total_size}", flush=True)
    recons = idct_1d_torch_dct(dct_vals)
    return recons


###############################################################################
# LOCAL TRAINING
###############################################################################
def train_for_steps(model, data_iter, local_steps, local_lr, prefix=""):
    device = model.fc1.weight.device
    rank = dist.get_rank()
    optimizer = optim.AdamW(model.parameters(), lr=local_lr)
    model.train()

    for step_idx in range(local_steps):
        x = next(data_iter)
        x = x.to(device)
        logits = model(x)
        loss = lm_loss_fn(logits, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step_idx+1) % 5 == 0:
            print(f"[Rank {rank}] {prefix} local_step={step_idx+1} loss={loss.item():.4f}", flush=True)


###############################################################################
# DILOCO INT8
###############################################################################
def diloco_int8(model, global_params_cpu, outer_opt_state,
                local_steps, data_iter, local_lr):
    """
    No DCT, just gather local delta => quant => average => broadcast => outer update
    """
    rank = dist.get_rank()
    device = model.fc1.weight.device
    world_size = dist.get_world_size()

    # Local training
    train_for_steps(model, data_iter, local_steps, local_lr, prefix="[diloco_int8]")

    # Compute delta
    with torch.no_grad():
        global_flat_gpu = global_params_cpu.view(-1).to(device)
        local_flat_gpu  = torch.cat([p.data.view(-1) for p in model.parameters()])
        delta_gpu = global_flat_gpu - local_flat_gpu
    delta_cpu = delta_gpu.cpu()

    # Quant + gather
    q_delta_cpu, scale, zero_point = quantize_int8(delta_cpu)
    scale_zero = torch.tensor([scale.item(), zero_point.item()], dtype=torch.float32)

    gather_qd = [torch.empty_like(q_delta_cpu) for _ in range(world_size)]
    gather_sz = [torch.empty_like(scale_zero)  for _ in range(world_size)]

    # measure & record bandwidth
    record_bandwidth(measure_all_gather(q_delta_cpu, world_size), rank)
    dist.all_gather(gather_qd, q_delta_cpu)
    dist.all_gather(gather_sz, scale_zero)

    # outer update on rank=0
    if rank == 0:
        all_deltas = []
        for i in range(world_size):
            sc = gather_sz[i][0]
            zp = gather_sz[i][1]
            deq = dequantize_int8(gather_qd[i], sc, zp)
            all_deltas.append(deq)
        stack_deltas = torch.stack(all_deltas, dim=0)
        mean_delta_cpu = stack_deltas.mean(dim=0)
    else:
        mean_delta_cpu = torch.empty_like(delta_cpu)

    # broadcast from rank=0
    record_bandwidth(measure_broadcast(mean_delta_cpu, world_size), rank)
    dist.broadcast(mean_delta_cpu, src=0)

    # Outer opt step
    outer_lr, outer_mom, outer_vel_cpu = outer_opt_state
    outer_vel_cpu = outer_mom * outer_vel_cpu + mean_delta_cpu
    new_global_params_cpu = global_params_cpu - outer_lr * outer_vel_cpu
    global_params_cpu.copy_(new_global_params_cpu)

    return global_params_cpu, (outer_lr, outer_mom, outer_vel_cpu)


###############################################################################
# DEMO INT8 (WITH DCT)
###############################################################################
def diloco_demo_int8(model, global_params_cpu, outer_opt_state,
                     local_steps, data_iter, local_lr, k_top):
    """
    Local training => compute local delta => DCT => keep top-k => quant => gather => iDCT => average => broadcast => outer update
    """
    rank = dist.get_rank()
    device = model.fc1.weight.device
    world_size = dist.get_world_size()

    # local training
    train_for_steps(model, data_iter, local_steps, local_lr, prefix="[demo_int8]")

    # compute delta
    with torch.no_grad():
        global_flat_gpu = global_params_cpu.view(-1).to(device)
        local_flat_gpu  = torch.cat([p.data.view(-1) for p in model.parameters()])
        delta_gpu = global_flat_gpu - local_flat_gpu
    delta_cpu = delta_gpu.cpu()

    # 1) DCT + top-k
    idx_cpu, amps_cpu = extract_fast_components_cpu(delta_cpu, k_top)

    # 2) quant + gather
    q_amps_cpu, scale, zero_point = quantize_int8(amps_cpu)
    scale_zero = torch.tensor([scale.item(), zero_point.item()], dtype=torch.float32)

    big_idx  = [torch.empty_like(idx_cpu)    for _ in range(world_size)]
    big_amps = [torch.empty_like(q_amps_cpu) for _ in range(world_size)]
    big_sz   = [torch.empty(2, dtype=torch.float32) for _ in range(world_size)]

    # measure & record bandwidth
    record_bandwidth(measure_all_gather(idx_cpu, world_size), rank)
    dist.all_gather(big_idx,  idx_cpu)

    record_bandwidth(measure_all_gather(q_amps_cpu, world_size), rank)
    dist.all_gather(big_amps, q_amps_cpu)

    record_bandwidth(measure_all_gather(scale_zero, world_size), rank)
    dist.all_gather(big_sz,   scale_zero)

    total_size = delta_cpu.numel()
    if rank == 0:
        recons_list = []
        for i in range(world_size):
            sc = big_sz[i][0]
            zp = big_sz[i][1]
            deq_amps_cpu = dequantize_int8(big_amps[i], sc, zp)
            partial_cpu = reconstruct_fast_components_cpu(big_idx[i], deq_amps_cpu, total_size)
            recons_list.append(partial_cpu)
        stack_recons = torch.stack(recons_list, dim=0)
        mean_recon_cpu = stack_recons.mean(dim=0)
    else:
        mean_recon_cpu = torch.empty(total_size, dtype=torch.float32)

    # broadcast
    record_bandwidth(measure_broadcast(mean_recon_cpu, world_size), rank)
    dist.broadcast(mean_recon_cpu, src=0)

    # 3) Outer update
    outer_lr, outer_mom, outer_vel_cpu = outer_opt_state
    outer_vel_cpu = outer_mom * outer_vel_cpu + mean_recon_cpu
    new_global_params_cpu = global_params_cpu - outer_lr * outer_vel_cpu
    global_params_cpu.copy_(new_global_params_cpu)

    return global_params_cpu, (outer_lr, outer_mom, outer_vel_cpu)


###############################################################################
# MAIN EXPERIMENT
###############################################################################
def run_experiment(rank, world_size, args):
    global cluster_bandwidth
    cluster_bandwidth = 0.0  # reset for each run

    setup_distributed(rank, world_size, backend=args.backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("[Rank 0] Using torch_dct for DCT/iDCT (no fallback).")
        print(f"[Rank 0] local_steps={args.local_steps}, num_outer_steps={args.num_outer_steps}")

    # Generate data with the "go up" pattern
    all_data = generate_synthetic_data(
        num_sequences=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=(42 + rank)
    )
    data_iter = infinite_data_loader(all_data, batch_size=args.batch_size, shuffle=True)

    # global model on CPU
    global_model = TinyLM(args.vocab_size, args.hidden_dim)
    global_params_cpu = torch.cat([p.data.view(-1) for p in global_model.parameters()])

    # local copy on GPU
    local_model = TinyLM(args.vocab_size, args.hidden_dim).to(device)
    with torch.no_grad():
        offset = 0
        for p in local_model.parameters():
            sz = p.numel()
            p.data.copy_(global_params_cpu[offset:offset+sz].view_as(p))
            offset += sz

    # Outer optimizer
    outer_lr = args.outer_lr
    outer_mom = args.outer_momentum
    outer_vel_cpu = torch.zeros_like(global_params_cpu)
    outer_opt_state = (outer_lr, outer_mom, outer_vel_cpu)

    if args.mode == 'diloco_int8':
        trainer_fn = lambda m, gp_cpu, ost: diloco_int8(
            m, gp_cpu, ost,
            args.local_steps,
            data_iter,
            args.inner_lr
        )
    else:
        trainer_fn = lambda m, gp_cpu, ost: diloco_demo_int8(
            m, gp_cpu, ost,
            args.local_steps,
            data_iter,
            args.inner_lr,
            k_top=args.k_top
        )

    for outer_step in range(args.num_outer_steps):
        if rank == 0:
            print(f"[Rank 0] Outer step={outer_step} about to call trainer_fn...", flush=True)
        global_params_cpu, outer_opt_state = trainer_fn(local_model, global_params_cpu, outer_opt_state)
        if rank == 0:
            print(f"[Rank 0] Outer step={outer_step} done trainer_fn, about to barrier...", flush=True)
        dist.barrier()
        if rank == 0:
            print(f"[Rank 0] Outer step={outer_step} done barrier", flush=True)

        if rank == 0 and (outer_step % args.print_every) == 0:
            print(f"[{args.mode}] rank=0, outer_step={outer_step}, "
                  f"current cluster_bandwidth={cluster_bandwidth/1e6:.3f} MB",
                  flush=True)

    if rank == 0:
        print(f"[Rank 0] all outer steps done, final barrier...", flush=True)
    dist.barrier()
    if rank == 0:
        print(f"[Rank 0] final barrier done. Now final eval if rank=0", flush=True)

    # Final eval
    if rank == 0:
        offset = 0
        for p in local_model.parameters():
            sz = p.numel()
            p.data.copy_(global_params_cpu[offset:offset+sz].view_as(p))
            offset += sz

        local_model.eval()
        test_loader = simple_data_loader(all_data, batch_size=args.batch_size, shuffle=False)
        total_loss = 0.0
        ccount = 0
        for batch in test_loader:
            with torch.no_grad():
                x = batch.to(device)
                logits = local_model(x)
                val = lm_loss_fn(logits, x).item()
                total_loss += val
                ccount += 1
        avg_loss = total_loss / max(ccount, 1)

        print(f"[Mode={args.mode}] rank=0 done. Final avg loss={avg_loss:.4f}")
        print(f"[Mode={args.mode}] rank=0 total cluster bandwidth={cluster_bandwidth/1e6:.3f} MB")

    cleanup_distributed()

def run_wrapped(rank, world_size, args):
    try:
        run_experiment(rank, world_size, args)
    except Exception as e:
        import traceback
        print(f"[Rank {rank}] error: {e}")
        print(traceback.format_exc())
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo','nccl'])
    parser.add_argument('--mode', type=str, default='diloco_int8',
                        choices=['diloco_int8', 'demo_int8'])

    parser.add_argument('--num_outer_steps', type=int, default=10)
    parser.add_argument('--local_steps', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-3)
    parser.add_argument('--outer_lr', type=float, default=0.1)
    parser.add_argument('--outer_momentum', type=float, default=0.9)
    parser.add_argument('--k_top', type=int, default=500)

    # Data + model
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)

    # Logging
    parser.add_argument('--print_every', type=int, default=1)
    args = parser.parse_args()

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    mp.spawn(run_wrapped, nprocs=args.world_size, join=True, args=(args.world_size, args))

if __name__ == "__main__":
    main()
