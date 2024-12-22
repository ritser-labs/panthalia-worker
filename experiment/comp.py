import torch
import torch.nn as nn
import torch.optim as optim
import math

###############################################################################
# Synthetic Dataset
###############################################################################

def make_synthetic_classification(n=20000, d=512, num_classes=10, seed=42):
    """
    Create a synthetic classification dataset:
      X in R^{n x d}, y in {0,...,num_classes-1}
    """
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    W_true = torch.randn(d, num_classes) * 0.1
    # Linear logits
    logits = X @ W_true
    y = logits.argmax(dim=1)
    return X, y

###############################################################################
# Simple MLP
###############################################################################

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def compute_loss(model, X, y):
    return nn.CrossEntropyLoss()(model(X), y)

def compute_accuracy(model, X, y):
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()

###############################################################################
# Compression Utilities: Top-K and Uniform Quantization
###############################################################################

def top_k_select(vec, k):
    """
    Returns indices and values of top-k by absolute magnitude.
    vec: 1D tensor
    """
    if k >= vec.numel():
        # take all
        idx = torch.arange(vec.numel(), device=vec.device)
        return idx, vec
    absvals = torch.abs(vec)
    threshold = torch.topk(absvals, k)[0][-1]
    mask = absvals >= threshold
    idx = mask.nonzero().squeeze()
    return idx, vec[idx]

def quantize_uniform(values, num_bits=8, eps=1e-9):
    """
    Uniform quantization to integers in [0, 2^num_bits -1]
    Returns qvals, (scale), (min_val) for dequant.
    """
    if len(values) == 0:
        return values, None, None
    vmin = values.min()
    vmax = values.max()
    rng = vmax - vmin
    if rng < eps:
        return torch.zeros_like(values), 1.0, float(vmin)
    max_int = (2**num_bits) - 1
    vals_norm = (values - vmin)/rng
    qvals = torch.round(vals_norm * max_int)
    return qvals, rng, float(vmin)

def dequantize_uniform(qvals, scale, vmin, num_bits=8):
    """
    Dequantize from [0,2^num_bits-1] back to real values.
    """
    if qvals is None or scale is None:
        return torch.tensor([], dtype=torch.float32, device=qvals.device)
    max_int = (2**num_bits) - 1
    return qvals*(scale/max_int) + vmin

###############################################################################
# 1) Fully Synchronous Baseline (No Compression)
###############################################################################

def run_baseline_sync(model, X, y, num_iters=100, lr=1e-3):
    """
    Two workers, each does a local step each iteration, then fully sync.
    No gradient/param compression.
    """
    device = X.device
    n = X.size(0)
    half = n // 2
    X0, y0 = X[:half], y[:half]
    X1, y1 = X[half:], y[half:]

    w0 = SimpleMLP(model.net[0].in_features, model.net[0].out_features,
                   model.net[-1].out_features).to(device)
    w1 = SimpleMLP(model.net[0].in_features, model.net[0].out_features,
                   model.net[-1].out_features).to(device)
    w0.load_state_dict(model.state_dict())
    w1.load_state_dict(model.state_dict())

    opt0 = optim.Adam(w0.parameters(), lr=lr)
    opt1 = optim.Adam(w1.parameters(), lr=lr)

    param_size = sum(p.numel() for p in w0.parameters())
    losses = []
    bits_per_iter = []

    for step in range(num_iters):
        # local step on worker0
        opt0.zero_grad()
        loss0 = compute_loss(w0, X0, y0)
        loss0.backward()
        opt0.step()

        # local step on worker1
        opt1.zero_grad()
        loss1 = compute_loss(w1, X1, y1)
        loss1.backward()
        opt1.step()

        # full param sync
        # param_size * 32 bits each direction => 2*(param_size*32)
        bw = 2*param_size*32
        bits_per_iter.append(bw)

        # average
        with torch.no_grad():
            for (p0, p1) in zip(w0.parameters(), w1.parameters()):
                avg = (p0.data + p1.data)/2
                p0.data.copy_(avg)
                p1.data.copy_(avg)

        losses.append(0.5*(loss0.item() + loss1.item()))

    return {
        'loss_curve': losses,
        'final_loss': losses[-1],
        'total_bits': sum(bits_per_iter),
        'model': w0
    }

###############################################################################
# 2) Fully Synchronous, but Compressed (Top-K + Quantization)
###############################################################################

def run_baseline_sync_compressed(model, X, y, num_iters=100, lr=1e-3,
                                 k=50000, num_bits=8):
    """
    Each iteration, 2 workers do local steps, then compress their entire param
    with top-k + uniform quant, average them.
    """
    device = X.device
    n = X.size(0)
    half = n // 2
    X0, y0 = X[:half], y[:half]
    X1, y1 = X[half:], y[half:]

    w0 = SimpleMLP(model.net[0].in_features, model.net[0].out_features,
                   model.net[-1].out_features).to(device)
    w1 = SimpleMLP(model.net[0].in_features, model.net[0].out_features,
                   model.net[-1].out_features).to(device)
    w0.load_state_dict(model.state_dict())
    w1.load_state_dict(model.state_dict())

    opt0 = optim.Adam(w0.parameters(), lr=lr)
    opt1 = optim.Adam(w1.parameters(), lr=lr)

    losses = []
    bits_per_iter = []

    def flatten_params(m):
        return torch.cat([p.data.view(-1) for p in m.parameters()])

    def unflatten_params(m, flat):
        offset = 0
        for p in m.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset+numel].view(p.size()))
            offset += numel

    for step in range(num_iters):
        # local step on each worker
        opt0.zero_grad()
        l0 = compute_loss(w0, X0, y0)
        l0.backward()
        opt0.step()

        opt1.zero_grad()
        l1 = compute_loss(w1, X1, y1)
        l1.backward()
        opt1.step()

        p0 = flatten_params(w0)
        p1 = flatten_params(w1)

        # top-k
        idx0, val0 = top_k_select(p0, k)
        idx1, val1 = top_k_select(p1, k)

        # quantize each
        qv0, rng0, m0 = quantize_uniform(val0, num_bits=num_bits)
        qv1, rng1, m1 = quantize_uniform(val1, num_bits=num_bits)

        # measure bandwidth: each selected param => 32 bits for index + num_bits for quant
        bw0 = len(val0)*(32 + num_bits)
        bw1 = len(val1)*(32 + num_bits)
        # overhead for storing scale + min => ~ 64 bits each => let's approximate 128
        bw0 += 128
        bw1 += 128
        bits_per_iter.append(bw0 + bw1)

        # dequant
        val0_deq = dequantize_uniform(qv0, rng0, m0, num_bits=num_bits)
        val1_deq = dequantize_uniform(qv1, rng1, m1, num_bits=num_bits)

        # reconstruct
        p0_recon = torch.zeros_like(p0)
        p1_recon = torch.zeros_like(p1)
        if len(val0)>0:
            p0_recon[idx0] = val0_deq
        if len(val1)>0:
            p1_recon[idx1] = val1_deq

        # average
        p_avg = 0.5*(p0_recon + p1_recon)

        # update each worker's params
        unflatten_params(w0, p_avg)
        unflatten_params(w1, p_avg)

        avg_loss = 0.5*(l0.item() + l1.item())
        losses.append(avg_loss)

    return {
        'loss_curve': losses,
        'final_loss': losses[-1],
        'total_bits': sum(bits_per_iter),
        'model': w0
    }

###############################################################################
# 3) DeMo-Like (Uncompressed)
###############################################################################

class DeMoWorker:
    def __init__(self, init_state, lr=1e-3, beta=0.9, device='cpu'):
        self.model = SimpleMLP(init_state['net.0.weight'].size(1),
                               init_state['net.0.weight'].size(0),
                               init_state['net.2.weight'].size(0)).to(device)
        self.model.load_state_dict(init_state)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.beta = beta
        self.momentum = None
        self.device = device

    def flatten_params(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def unflatten_params(self, flat):
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset+numel].view(p.size()))
            offset += numel

    def flatten_momentum(self):
        if self.momentum is None:
            self.momentum = torch.zeros_like(self.flatten_params())
        return self.momentum

def run_demo_like(
    model, X, y,
    num_outer=10, H=5, k=10000,
    lr=1e-3, beta=0.9,
    device='cpu'
):
    """
    Each outer step => H local steps, then partial momentum sync (uncompressed).
    """
    X, y = X.to(device), y.to(device)
    n = X.size(0)
    half = n//2
    X0,y0 = X[:half],y[:half]
    X1,y1 = X[half:],y[half:]

    init_state = model.state_dict()
    w0 = DeMoWorker(init_state, lr=lr, beta=beta, device=device)
    w1 = DeMoWorker(init_state, lr=lr, beta=beta, device=device)

    param_size = sum(p.numel() for p in w0.model.parameters())
    bits_list = []
    loss_curve = []

    def local_train(worker, x_, y_):
        worker.opt.zero_grad()
        loss = compute_loss(worker.model, x_, y_)
        loss.backward()
        worker.opt.step()
        return loss.item()

    def update_momentum(worker, oldp, newp):
        if worker.momentum is None:
            worker.momentum = torch.zeros_like(oldp)
        diff = newp - oldp
        worker.momentum = beta*worker.momentum + diff

    for outer_step in range(num_outer):
        oldp0 = w0.flatten_params()
        oldp1 = w1.flatten_params()

        local_loss0 = 0
        local_loss1 = 0
        for _ in range(H):
            local_loss0 += local_train(w0, X0, y0)
            local_loss1 += local_train(w1, X1, y1)

        newp0 = w0.flatten_params()
        newp1 = w1.flatten_params()

        update_momentum(w0, oldp0, newp0)
        update_momentum(w1, oldp1, newp1)

        # top-k momentum sync (NO quant)
        mom0 = w0.momentum
        mom1 = w1.momentum

        idx0, val0 = top_k_select(mom0, k)
        idx1, val1 = top_k_select(mom1, k)

        # each selected => 32 bits index + 32 bits value
        bw0 = len(val0)*(32+32)
        bw1 = len(val1)*(32+32)
        bits_list.append(bw0 + bw1)

        # reconstruct fast portion
        fast0 = torch.zeros_like(mom0)
        fast1 = torch.zeros_like(mom1)
        if len(val0)>0:
            fast0[idx0] = val0
        if len(val1)>0:
            fast1[idx1] = val1

        fast_avg = 0.5*(fast0 + fast1)

        # param update => param = param - fast_avg
        w0.unflatten_params(w0.flatten_params() - fast_avg)
        w1.unflatten_params(w1.flatten_params() - fast_avg)

        # remove fast portion from local momentum
        w0.momentum = mom0 - fast0
        w1.momentum = mom1 - fast1

        avg_loss = 0.5*(local_loss0 + local_loss1)/H
        loss_curve.append(avg_loss)

    final_loss = 0.5*(compute_loss(w0.model, X, y).item() +
                      compute_loss(w1.model, X, y).item())
    return {
        'loss_curve': loss_curve,
        'final_loss': final_loss,
        'total_bits': sum(bits_list),
        'model': w0.model,
        'model2': w1.model
    }

###############################################################################
# 4) DeMo-Like with Top-K + Quant on Momentum
###############################################################################

def run_demo_like_compressed(
    model, X, y,
    num_outer=10, H=5, k=10000, num_bits=8,
    lr=1e-3, beta=0.9,
    device='cpu'
):
    """
    Each outer step => H local steps, then partial momentum sync (top-k + quant).
    """
    X, y = X.to(device), y.to(device)
    n = X.size(0)
    half = n//2
    X0,y0 = X[:half],y[:half]
    X1,y1 = X[half:], y[half:]

    init_state = model.state_dict()
    w0 = DeMoWorker(init_state, lr=lr, beta=beta, device=device)
    w1 = DeMoWorker(init_state, lr=lr, beta=beta, device=device)

    param_size = sum(p.numel() for p in w0.model.parameters())
    bits_list = []
    loss_curve = []

    def local_train(worker, x_, y_):
        worker.opt.zero_grad()
        loss = compute_loss(worker.model, x_, y_)
        loss.backward()
        worker.opt.step()
        return loss.item()

    def update_momentum(worker, oldp, newp):
        if worker.momentum is None:
            worker.momentum = torch.zeros_like(oldp)
        diff = newp - oldp
        worker.momentum = beta*worker.momentum + diff

    for outer_step in range(num_outer):
        oldp0 = w0.flatten_params()
        oldp1 = w1.flatten_params()

        local_loss0 = 0
        local_loss1 = 0
        for _ in range(H):
            local_loss0 += local_train(w0, X0, y0)
            local_loss1 += local_train(w1, X1, y1)

        newp0 = w0.flatten_params()
        newp1 = w1.flatten_params()

        # momentum update
        update_momentum(w0, oldp0, newp0)
        update_momentum(w1, oldp1, newp1)

        mom0 = w0.momentum
        mom1 = w1.momentum

        # top-k
        idx0, val0 = top_k_select(mom0, k)
        idx1, val1 = top_k_select(mom1, k)

        # quantize each
        qv0, rng0, m0 = quantize_uniform(val0, num_bits=num_bits)
        qv1, rng1, m1 = quantize_uniform(val1, num_bits=num_bits)

        # bits = (#val0*(32+num_bits)) + (#val1*(32+num_bits)) + overhead
        bw0 = len(val0)*(32+num_bits) + 128
        bw1 = len(val1)*(32+num_bits) + 128
        bits_list.append(bw0+bw1)

        # dequant
        val0_deq = dequantize_uniform(qv0, rng0, m0, num_bits=num_bits)
        val1_deq = dequantize_uniform(qv1, rng1, m1, num_bits=num_bits)

        # reconstruct fast portion
        fast0 = torch.zeros_like(mom0)
        fast1 = torch.zeros_like(mom1)
        if len(val0_deq)>0:
            fast0[idx0] = val0_deq
        if len(val1_deq)>0:
            fast1[idx1] = val1_deq

        fast_avg = 0.5*(fast0 + fast1)

        # outer param update => param -= fast_avg
        w0.unflatten_params(w0.flatten_params() - fast_avg)
        w1.unflatten_params(w1.flatten_params() - fast_avg)

        # remove fast portion from momentum
        w0.momentum = mom0 - fast0
        w1.momentum = mom1 - fast1

        avg_loss = 0.5*(local_loss0 + local_loss1)/H
        loss_curve.append(avg_loss)

    final_loss = 0.5*(compute_loss(w0.model, X, y).item() +
                      compute_loss(w1.model, X, y).item())
    return {
        'loss_curve': loss_curve,
        'final_loss': final_loss,
        'total_bits': sum(bits_list),
        'model': w0.model,
        'model2': w1.model
    }

###############################################################################
# MAIN
###############################################################################

def main():
    device = 'cpu'  # or 'cuda'
    X, y = make_synthetic_classification(n=20000, d=512, num_classes=10, seed=123)
    X, y = X.to(device), y.to(device)

    base_model = SimpleMLP(512, 256, 10).to(device)

    print("Running baseline (no compression):")
    r_base = run_baseline_sync(base_model, X, y, num_iters=50, lr=1e-3)
    print(f"  => final_loss={r_base['final_loss']:.4f}, total_bits={r_base['total_bits']:.0f}")

    print("\nRunning baseline with top-k + quant:")
    r_cbase = run_baseline_sync_compressed(base_model, X, y, num_iters=50, lr=1e-3,
                                           k=50000, num_bits=8)
    print(f"  => final_loss={r_cbase['final_loss']:.4f}, total_bits={r_cbase['total_bits']:.0f}")

    print("\nRunning DeMo-like (uncompressed):")
    r_demo = run_demo_like(base_model, X, y, num_outer=10, H=5, k=20000, lr=1e-3, beta=0.9)
    print(f"  => final_loss={r_demo['final_loss']:.4f}, total_bits={r_demo['total_bits']:.0f}")

    print("\nRunning DeMo-like with top-k + quant on momentum:")
    r_cdemo = run_demo_like_compressed(base_model, X, y,
                                       num_outer=10, H=5, k=20000, num_bits=8,
                                       lr=1e-3, beta=0.9)
    print(f"  => final_loss={r_cdemo['final_loss']:.4f}, total_bits={r_cdemo['total_bits']:.0f}")

    # Optionally measure final accuracy
    acc_base = compute_accuracy(r_base['model'], X, y)
    acc_cbase = compute_accuracy(r_cbase['model'], X, y)
    acc_demo = 0.5*(compute_accuracy(r_demo['model'], X, y) +
                    compute_accuracy(r_demo['model2'], X, y))
    acc_cdemo = 0.5*(compute_accuracy(r_cdemo['model'], X, y) +
                     compute_accuracy(r_cdemo['model2'], X, y))

    print("\n=== Final Accuracies ===")
    print(f"Baseline No-Compress: {acc_base:.4f}")
    print(f"Baseline Compressed : {acc_cbase:.4f}")
    print(f"DeMo Uncompressed   : {acc_demo:.4f}")
    print(f"DeMo Compressed     : {acc_cdemo:.4f}")

if __name__=="__main__":
    main()
