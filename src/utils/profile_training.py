import time
from collections import defaultdict

import torch


def profile_training(model, optimizer, train_loader, device, heat_loss, loading_loss, heat_std, load_std,
                     n_batches=100, warmup=10, profile_ops=True):
    cuda = torch.cuda.is_available() and str(device).startswith('cuda')
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    it, n_g, t0 = iter(train_loader), 0, time.perf_counter()
    for i in range(warmup + n_batches):
        try:
            data = next(it)
        except StopIteration:
            break
        if i == warmup:
            n_g, t0 = 0, time.perf_counter()
        n_g += data.num_graphs
    loader_gps = n_g / (time.perf_counter() - t0)
    print(f'[1] data loader alone: {loader_gps:.0f} graphs/s')

    model.train()
    t = defaultdict(float)
    n_done, n_g = 0, 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    it = iter(train_loader)
    for i in range(warmup + n_batches):
        sync(); a = time.perf_counter()
        try:
            data = next(it)
        except StopIteration:
            break
        b = time.perf_counter()
        data = data.to(device, non_blocking=True); sync(); c = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        pred, extras, _, _, aux, _, _, _, _ = model(data); sync(); d = time.perf_counter()
        N_model = extras[5]
        l_heat = heat_loss(pred, data.y, heat_std)
        l_load = loading_loss(N_model, data.y_load, load_std)
        l_aux = sum(aux)
        loss = l_heat + l_load + l_aux; sync(); e = time.perf_counter()
        loss.backward(); sync(); f = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step(); sync(); g = time.perf_counter()
        _ = [v.item() for v in (loss, l_heat, l_load, l_aux)]
        _ = (pred - data.y).abs().sum(0).cpu(), (N_model - data.y_load).abs().sum(0).cpu()
        sync(); h = time.perf_counter()
        if i >= warmup:
            for k, (x, y) in {'data wait': (a, b), 'to(device)': (b, c), 'forward': (c, d), 'loss': (d, e),
                              'backward': (e, f), 'clip + step': (f, g), 'logging (.item/.cpu)': (g, h)}.items():
                t[k] += y - x
            n_done += 1
            n_g += data.num_graphs

    total = sum(t.values())
    print(f'[2] training step, {n_done} batches (after {warmup} warm-up), {n_g / total:.0f} graphs/s')
    for k, v in t.items():
        print(f'    {k:22s} {1e3 * v / n_done:8.2f} ms/batch  {100 * v / total:5.1f} %')
    if cuda:
        print(f'    peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB')
    if t['data wait'] / total > 0.2:
        print('    -> data-bound: more num_workers, larger prefetch, or lighter collate')

    if profile_ops:
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        data = next(iter(train_loader)).to(device)
        with profile(activities=acts, record_shapes=False) as prof:
            for _ in range(5):
                optimizer.zero_grad(set_to_none=True)
                pred, extras, _, _, aux, _, _, _, _ = model(data)
                loss = heat_loss(pred, data.y, heat_std) + loading_loss(extras[5], data.y_load, load_std) + sum(aux)
                loss.backward()
            sync()
        key = 'cuda_time_total' if cuda else 'cpu_time_total'
        print(f'[3] top ops by {key} (5 steps on one batch)')
        print(prof.key_averages().table(sort_by=key, row_limit=15))
    return dict(t), loader_gps
