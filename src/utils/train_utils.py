import torch
import numpy as np
import time
from tqdm import tqdm
from .utils import compute_grad_similarity, compute_grad_from_init, soft_modularity_batch, effective_num_clusters_batch, PressureMetrics, HeatMetrics


def regression_metrics(pred, target, names):
    out = {}
    err = pred - target
    for i, n in enumerate(names):
        e, t = err[:, i], target[:, i]
        out[f"rmse/{n}"] = float(np.sqrt((e ** 2).mean()))
        out[f"r2/{n}"] = float(1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum())
    out["rmse"] = float(np.sqrt((err ** 2).mean()))
    out["r2"] = float(np.mean([out[f"r2/{n}"] for n in names]))
    if pred.shape[1] > 1:
        dp = pred - pred.mean(1, keepdims=True)
        dt = target - target.mean(1, keepdims=True)
        out["dev_mae"] = float(np.abs(dp - dt).mean())
        out["dev_std_true"] = float(dt.std())
        out["dev_r2"] = float(1 - ((dp - dt) ** 2).sum() / ((dt - dt.mean()) ** 2).sum())
        out["mean_mae"] = float(np.abs(pred.mean(1) - target.mean(1)).mean())
    return out


def get_target_names(loader, n_out):
    ds = loader.dataset
    ds = getattr(ds, "dataset", ds)
    names = getattr(ds, "target_name", None)
    return list(names) if names is not None else [f"t{i}" for i in range(n_out)]


def heat_loss(Q_pred, y_heat, std):
    return ((Q_pred - y_heat).abs() / std).mean()


def loading_loss(N_model, y_load, std):
    return ((N_model - y_load).abs() / std).mean()


def train_base(model, optimizer, scheduler, criterion, best_model_path, train_loader, valid_loader, device, args):
    if args.wandb:
        import wandb

    train_length = len(train_loader.dataset)
    valid_length = len(valid_loader.dataset)

    best_valid_mae = float('inf')
    best_valid_rmse = float('inf')
    best_valid_r2 = 0.0
    best_metrics = {}
    target_names = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_task_loss = 0.0

        for data in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)

            target = data.y
            task_loss = criterion(pred, target)
            task_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            train_task_loss += task_loss.item() * data.num_graphs

        train_task_loss /= train_length

        scheduler.step()

        model.eval()
        valid_task_loss = 0.0
        preds, targets = [], []

        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]"):
                data = data.to(device)
                pred = model(data)

                target = data.y
                task_loss = criterion(pred, target)

                valid_task_loss += task_loss.item() * data.num_graphs
                preds.append(pred.cpu())
                targets.append(target.cpu())

        valid_task_loss /= valid_length
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        if target_names is None:
            target_names = get_target_names(valid_loader, preds.shape[1])
        m = regression_metrics(preds, targets, target_names)
        valid_rmse, valid_r2 = m["rmse"], m["r2"]

        if valid_task_loss < best_valid_mae:
            best_valid_mae = valid_task_loss
            best_valid_rmse = valid_rmse
            best_valid_r2 = valid_r2
            best_metrics = dict(m, epoch=epoch)
            torch.save(model.state_dict(), best_model_path)

        per_p = " ".join(f"{n.split('_')[-1]}:{m[f'mae/{n}']:.3f}" for n in target_names)
        dev = f" | dev MAE: {m['dev_mae']:.3f} (true std {m['dev_std_true']:.3f}), dev R2: {m['dev_r2']:.3f}" if "dev_mae" in m else ""
        print(f"Epoch {epoch:03d}: Train MAE: {train_task_loss:.4f} | Valid MAE: {valid_task_loss:.4f} "
            f"[{per_p}]{dev} | LR: {scheduler.get_last_lr()[0]:.4e} |")

        row = {"epoch": epoch,
               "train_mae": train_task_loss,
               "valid_mae": valid_task_loss,
               "lr": scheduler.get_last_lr()[0],
               **{f"valid_{k}": v for k, v in m.items()}}
        if args.wandb:
            wandb.log(row)

    print(f"Best model with Valid MAE: {best_valid_mae:.4f} (epoch {best_metrics.get('epoch')})")
    for n in target_names:
        print(f"  {n}: MAE {best_metrics[f'mae/{n}']:.4f}, RMSE {best_metrics[f'rmse/{n}']:.4f}, R2 {best_metrics[f'r2/{n}']:.4f}")
    if "dev_mae" in best_metrics:
        print(f"  pressure component: dev MAE {best_metrics['dev_mae']:.4f} "
            f"(true std {best_metrics['dev_std_true']:.4f}), dev R2 {best_metrics['dev_r2']:.4f}")
        print(f"  structure component: mean MAE {best_metrics['mean_mae']:.4f}")

    if args.wandb:
        wandb.log({"best_valid_mae": best_valid_mae,
                "best_valid_rmse": best_valid_rmse,
                "best_valid_r2": best_valid_r2,
                **{f"best_valid_{k}": v for k, v in best_metrics.items()}})


def train_base_heat(model, optimizer, scheduler, criterion, best_model_path, train_loader, valid_loader, device, args):
    if args.wandb:
        import wandb
        wandb.define_metric('epoch')
        wandb.define_metric('*', step_metric='epoch')

    heat_std = torch.tensor(args.heat_std, dtype=torch.float, device=device)
    train_m = HeatMetrics(args.pressures, heat_std)
    valid_m = HeatMetrics(args.pressures, heat_std)
    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)

    best = {'heat': float('inf'), 'epoch': None}
    heat_names = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_m.reset()
        tr = {'total': 0., 'heat': 0.}
        for data in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            assert pred.shape == data.y.shape, f'pred {tuple(pred.shape)} vs target {tuple(data.y.shape)}'

            l_heat = heat_loss(pred, data.y, heat_std)
            loss = l_heat
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            b = data.num_graphs
            for k, v in zip(tr, (loss, l_heat)):
                tr[k] += v.item() * b
            train_m.update(pred, data.y)
        scheduler.step()
        tr = {k: v / n_train for k, v in tr.items()}

        model.eval()
        valid_m.reset()
        va = {'total': 0., 'heat': 0.}
        q_pred, q_true = [], []
        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f'Epoch {epoch} [Valid]'):
                data = data.to(device)
                pred = model(data)

                l_heat = heat_loss(pred, data.y, heat_std)
                b = data.num_graphs
                for k, v in zip(va, (l_heat, l_heat)):
                    va[k] += v.item() * b
                valid_m.update(pred, data.y)

                q_pred.append(pred.cpu()); q_true.append(data.y.cpu())
        va = {k: v / n_valid for k, v in va.items()}

        q_pred, q_true = torch.cat(q_pred).numpy(), torch.cat(q_true).numpy()
        if heat_names is None:
            heat_names = get_target_names(valid_loader, q_pred.shape[1])
        reg = {'heat': regression_metrics(q_pred, q_true, heat_names)}

        ts, vs = train_m.summary(), valid_m.summary()
        lr = scheduler.get_last_lr()[0]

        if vs['heat'] < best['heat']:
            best = {**vs, 'epoch': epoch,
                    **{f'{k}@{p:g}': v[j].item() for k, v in valid_m.compute().items() for j, p in enumerate(args.pressures)},
                    **{f'heat_{k}': reg['heat'][k] for k in args.reg_keys if k in reg['heat']},
                    **{f'heat_r2@{p:g}': reg['heat'][f'r2/{n}'] for p, n in zip(args.pressures, heat_names)}}
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch:03d} | train heat {ts['heat']:.3f} "
              f"| valid heat {vs['heat']:.3f} (nmae {vs['heat_nmae']:.3f}, R2 {reg['heat']['r2']:.3f}) | lr {lr:.2e}")
        print(f"          valid heat [{valid_m.line('heat')}] "
              f"| heat dev R2 {reg['heat'].get('dev_r2', float('nan')):.3f}")

        if args.wandb:
            row = {'epoch': epoch, 'train/lr': lr,
                   **valid_m.wandb_sections(args.gas),
                   **{f'train/loss_{k}': v for k, v in tr.items()},
                   **{f'train/{k}': v for k, v in ts.items()},
                   **{f'valid/loss_{k}': v for k, v in va.items()},
                   **{f'valid/heat_{k}': reg['heat'][k] for k in args.reg_keys if k in reg['heat']}}
            wandb.log(row)

    print(f"Best model: epoch {best['epoch']}")
    print(f"  heat: MAE {best['heat']:.4f} (nmae {best['heat_nmae']:.4f}) | RMSE {best['heat_rmse']:.4f} | "
          f"R2 {best['heat_r2']:.4f} | dev R2 {best.get('heat_dev_r2', float('nan')):.4f} "
          f"(true dev std {reg['heat'].get('dev_std_true', float('nan')):.3f}) | mean MAE {best.get('heat_mean_mae', float('nan')):.4f}")
    for p in args.pressures:
        print(f"  {p:g} bar: heat {best[f'heat@{p:g}']:.4f} (nmae {best[f'heat_nmae@{p:g}']:.4f}, R2 {best[f'heat_r2@{p:g}']:.4f})")
    if args.wandb:
        wandb.run.summary.update({f'best/{k}': v for k, v in best.items()})


def train_base_heat_load(model, optimizer, scheduler, criterion, best_model_path, train_loader, valid_loader, device, args):
    if args.wandb:
        import wandb
        wandb.define_metric('epoch')
        wandb.define_metric('*', step_metric='epoch')

    heat_std = torch.tensor(args.heat_std, dtype=torch.float, device=device)
    load_std = torch.tensor(args.load_std, dtype=torch.float, device=device)
    train_m = PressureMetrics(args.pressures, heat_std, load_std)
    valid_m = PressureMetrics(args.pressures, heat_std, load_std)
    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)

    best = {'heat': float('inf'), 'epoch': None}
    heat_names = None
    load_names = [f'load_{args.gas}_{p:g}' for p in args.pressures]

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_m.reset()
        tr = {'total': 0., 'heat': 0., 'load': 0.}
        for data in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            data = data.to(device)
            optimizer.zero_grad()
            pred, N_model = model(data)
            assert pred.shape == data.y.shape, f'pred {tuple(pred.shape)} vs target {tuple(data.y.shape)}'
            assert N_model.shape == data.y_load.shape, f'load {tuple(N_model.shape)} vs target {tuple(data.y_load.shape)}'

            l_heat = heat_loss(pred, data.y, heat_std)
            l_load = loading_loss(N_model, data.y_load, load_std)
            loss = l_heat + l_load
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            b = data.num_graphs
            for k, v in zip(tr, (loss, l_heat, l_load)):
                tr[k] += v.item() * b
            train_m.update(pred, data.y, N_model, data.y_load)
        scheduler.step()
        tr = {k: v / n_train for k, v in tr.items()}

        model.eval()
        valid_m.reset()
        va = {'total': 0., 'heat': 0., 'load': 0.}
        q_pred, q_true, n_pred, n_true = [], [], [], []
        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f'Epoch {epoch} [Valid]'):
                data = data.to(device)
                pred, N_model = model(data)

                l_heat = heat_loss(pred, data.y, heat_std)
                l_load = loading_loss(N_model, data.y_load, load_std)
                b = data.num_graphs
                for k, v in zip(va, (l_heat + l_load, l_heat, l_load)):
                    va[k] += v.item() * b
                valid_m.update(pred, data.y, N_model, data.y_load)

                q_pred.append(pred.cpu()); q_true.append(data.y.cpu())
                n_pred.append(N_model.cpu()); n_true.append(data.y_load.cpu())
        va = {k: v / n_valid for k, v in va.items()}

        q_pred, q_true = torch.cat(q_pred).numpy(), torch.cat(q_true).numpy()
        n_pred, n_true = torch.cat(n_pred).numpy(), torch.cat(n_true).numpy()
        if heat_names is None:
            heat_names = get_target_names(valid_loader, q_pred.shape[1])
        reg = {'heat': regression_metrics(q_pred, q_true, heat_names),
               'load': regression_metrics(n_pred, n_true, load_names)}

        ts, vs = train_m.summary(), valid_m.summary()
        lr = scheduler.get_last_lr()[0]

        if vs['heat'] < best['heat']:
            best = {**vs, 'epoch': epoch,
                    **{f'{k}@{p:g}': v[j].item() for k, v in valid_m.compute().items() for j, p in enumerate(args.pressures)},
                    **{f'{t}_{k}': reg[t][k] for t in reg for k in args.reg_keys if k in reg[t]},
                    **{f'{t}_r2@{p:g}': reg[t][f'r2/{n}'] for t, names in (('heat', heat_names), ('load', load_names))
                       for p, n in zip(args.pressures, names)}}
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch:03d} | train heat {ts['heat']:.3f} load {ts['load']:.4f} "
              f"| valid heat {vs['heat']:.3f} (nmae {vs['heat_nmae']:.3f}, R2 {reg['heat']['r2']:.3f}) "
              f"load {vs['load']:.4f} (nmae {vs['load_nmae']:.3f}, R2 {reg['load']['r2']:.3f}) | lr {lr:.2e}")
        print(f"          valid heat [{valid_m.line('heat')}] | load nmae [{valid_m.line('load_nmae')}] "
              f"| heat dev R2 {reg['heat'].get('dev_r2', float('nan')):.3f}")

        if args.wandb:
            row = {'epoch': epoch, 'train/lr': lr,
                   **valid_m.wandb_sections(args.gas),
                   **{f'train/loss_{k}': v for k, v in tr.items()},
                   **{f'train/{k}': v for k, v in ts.items()},
                   **{f'valid/loss_{k}': v for k, v in va.items()},
                   **{f'valid/{t}_{k}': reg[t][k] for t in reg for k in args.reg_keys if k in reg[t]}}
            wandb.log(row)

    print(f"Best model: epoch {best['epoch']}")
    print(f"  heat: MAE {best['heat']:.4f} (nmae {best['heat_nmae']:.4f}) | RMSE {best['heat_rmse']:.4f} | "
          f"R2 {best['heat_r2']:.4f} | dev R2 {best.get('heat_dev_r2', float('nan')):.4f} "
          f"(true dev std {reg['heat'].get('dev_std_true', float('nan')):.3f}) | mean MAE {best.get('heat_mean_mae', float('nan')):.4f}")
    print(f"  load: MAE {best['load']:.4f} (nmae {best['load_nmae']:.4f}) | RMSE {best['load_rmse']:.4f} | "
          f"R2 {best['load_r2']:.4f} | dev R2 {best.get('load_dev_r2', float('nan')):.4f} "
          f"(true dev std {reg['load'].get('dev_std_true', float('nan')):.3f}) | mean MAE {best.get('load_mean_mae', float('nan')):.4f}")
    for p in args.pressures:
        print(f"  {p:g} bar: heat {best[f'heat@{p:g}']:.4f} (nmae {best[f'heat_nmae@{p:g}']:.4f}, R2 {best[f'heat_r2@{p:g}']:.4f}) | "
              f"load {best[f'load@{p:g}']:.4f} (nmae {best[f'load_nmae@{p:g}']:.4f}, R2 {best[f'load_r2@{p:g}']:.4f})")
    if args.wandb:
        wandb.run.summary.update({f'best/{k}': v for k, v in best.items()})


def train_pool(model, optimizer, scheduler, criterion, best_model_path, train_loader, valid_loader, device, args):
    if args.wandb:
        import wandb

    train_length = len(train_loader.dataset)
    valid_length = len(valid_loader.dataset)

    best_valid_mae = float('inf')
    best_valid_rmse = float('inf')
    best_valid_r2 = 0.0
    best_metrics = {}
    target_names = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_task_loss = 0.0
        train_aux_loss = 0.0

        for data in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()
            pred, m, aux, s, _, _, _ = model(data)

            target = data.y
            if args.readout == "add":
                pred = pred.sum(dim=1)
            elif args.readout == "mean":
                pred = pred.sum(dim=1) / m.sum(dim=1)
            else:
                raise ValueError(f"unknown readout: {args.readout}")
            assert pred.shape == target.shape, f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            task_loss = criterion(pred, target)
            aux_loss = sum(aux)
            total_loss = task_loss + aux_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            train_total_loss += total_loss.item() * data.num_graphs
            train_task_loss += task_loss.item() * data.num_graphs
            train_aux_loss += aux_loss.item() * data.num_graphs

        train_total_loss /= train_length
        train_task_loss /= train_length
        train_aux_loss /= train_length

        scheduler.step()

        model.eval()
        valid_total_loss = 0.0
        valid_task_loss = 0.0
        valid_aux_loss = 0.0
        valid_mod = [0. for _ in range(args.pool_num)]
        valid_k_eff = [0. for _ in range(args.pool_num)]
        valid_na = [0. for _ in range(args.pool_num)]
        preds, targets = [], []

        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]"):
                data = data.to(device)
                pred, m, aux, s, adj, na, _ = model(data)

                target = data.y
                if args.readout == "add":
                    pred = pred.sum(dim=1)
                elif args.readout == "mean":
                    pred = pred.sum(dim=1) / m.sum(dim=1)
                else:
                    print("readout error")
                    raise ValueError(f"unknown readout: {args.readout}")
                task_loss = criterion(pred, target)
                aux_loss = sum(aux)
                total_loss = task_loss + aux_loss

                valid_total_loss += total_loss.item() * data.num_graphs
                valid_task_loss += task_loss.item() * data.num_graphs
                valid_aux_loss += aux_loss.item() * data.num_graphs
                for i in range(args.pool_num):
                    valid_mod[i] += soft_modularity_batch(adj[i], s[i])
                    valid_k_eff[i] += effective_num_clusters_batch(s[i])
                    valid_na[i] += na[i].sum().item()
                preds.append(pred.cpu())
                targets.append(target.cpu())

        valid_total_loss /= valid_length
        valid_task_loss /= valid_length
        valid_aux_loss /= valid_length
        for i in range(args.pool_num):
            valid_mod[i] /= valid_length
            valid_k_eff[i] /= valid_length
            valid_na[i] /= valid_length
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        if target_names is None:
            target_names = get_target_names(valid_loader, preds.shape[1])
        m = regression_metrics(preds, targets, target_names)
        valid_rmse, valid_r2 = m["rmse"], m["r2"]

        if valid_task_loss < best_valid_mae:
            best_valid_mae = valid_task_loss
            best_valid_rmse = valid_rmse
            best_valid_r2 = valid_r2
            best_metrics = dict(m, epoch=epoch)
            torch.save(model.state_dict(), best_model_path)

        per_p = " ".join(f"{n.split('_')[-1]}:{m[f'mae/{n}']:.3f}" for n in target_names)
        dev = f" | dev MAE: {m['dev_mae']:.3f} (true std {m['dev_std_true']:.3f}), dev R2: {m['dev_r2']:.3f}" if "dev_mae" in m else ""
        print(f"Epoch {epoch:03d}: Train MAE: {train_task_loss:.4f} | Valid MAE: {valid_task_loss:.4f} "
            f"[{per_p}]{dev} | LR: {scheduler.get_last_lr()[0]:.4e} |")

        row = {"epoch": epoch,
               "train_mae": train_task_loss,
               "train_aux": train_aux_loss,
               "valid_mae": valid_task_loss,
               "valid_aux": valid_aux_loss,
               "lr": scheduler.get_last_lr()[0],
               **{f"valid_{k}": v for k, v in m.items()}}
        for i in range(args.pool_num):
            row[f"valid_mod{i+1}"] = valid_mod[i].item()
            row[f"valid_k_eff{i+1}"] = valid_k_eff[i].item()
            row[f"valid_na{i+1}"] = valid_na[i]
        if args.wandb:
            wandb.log(row)

    print(f"Best model with Valid MAE: {best_valid_mae:.4f} (epoch {best_metrics.get('epoch')})")
    for n in target_names:
        print(f"  {n}: MAE {best_metrics[f'mae/{n}']:.4f}, RMSE {best_metrics[f'rmse/{n}']:.4f}, R2 {best_metrics[f'r2/{n}']:.4f}")
    if "dev_mae" in best_metrics:
        print(f"  pressure component: dev MAE {best_metrics['dev_mae']:.4f} "
            f"(true std {best_metrics['dev_std_true']:.4f}), dev R2 {best_metrics['dev_r2']:.4f}")
        print(f"  structure component: mean MAE {best_metrics['mean_mae']:.4f}")

    if args.wandb:
        wandb.log({"best_valid_mae": best_valid_mae,
                "best_valid_rmse": best_valid_rmse,
                "best_valid_r2": best_valid_r2,
                **{f"best_valid_{k}": v for k, v in best_metrics.items()}})


def train_pool_attn(model, optimizer, scheduler, criterion, best_model_path, train_loader, valid_loader, device, args):
    if args.wandb:
        import wandb

    train_length = len(train_loader.dataset)
    valid_length = len(valid_loader.dataset)

    best_valid_mae = float('inf')
    best_valid_rmse = float('inf')
    best_valid_r2 = 0.0
    best_metrics = {}
    target_names = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_task_loss = 0.0
        train_aux_loss = 0.0

        for data in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()
            pred, e, m, alpha, aux, s, _, _, _ = model(data)

            target = data.y
            assert pred.shape == target.shape, f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            task_loss = criterion(pred, target)
            aux_loss = sum(aux)
            total_loss = task_loss + aux_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            train_total_loss += total_loss.item() * data.num_graphs
            train_task_loss += task_loss.item() * data.num_graphs
            train_aux_loss += aux_loss.item() * data.num_graphs

        train_total_loss /= train_length
        train_task_loss /= train_length
        train_aux_loss /= train_length

        scheduler.step()

        model.eval()
        valid_total_loss = 0.0
        valid_task_loss = 0.0
        valid_aux_loss = 0.0
        valid_mod = [0. for _ in range(args.pool_num)]
        valid_k_eff = [0. for _ in range(args.pool_num)]
        valid_na = [0. for _ in range(args.pool_num)]
        preds, targets = [], []

        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]"):
                data = data.to(device)
                pred, e, m, alpha, aux, s, adj, na, _ = model(data)

                target = data.y
                task_loss = criterion(pred, target)
                aux_loss = sum(aux)
                total_loss = task_loss + aux_loss

                valid_total_loss += total_loss.item() * data.num_graphs
                valid_task_loss += task_loss.item() * data.num_graphs
                valid_aux_loss += aux_loss.item() * data.num_graphs
                for i in range(args.pool_num):
                    valid_mod[i] += soft_modularity_batch(adj[i], s[i])
                    valid_k_eff[i] += effective_num_clusters_batch(s[i])
                    valid_na[i] += na[i].sum().item()
                preds.append(pred.cpu())
                targets.append(target.cpu())

        valid_total_loss /= valid_length
        valid_task_loss /= valid_length
        valid_aux_loss /= valid_length
        for i in range(args.pool_num):
            valid_mod[i] /= valid_length
            valid_k_eff[i] /= valid_length
            valid_na[i] /= valid_length
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        if target_names is None:
            target_names = get_target_names(valid_loader, preds.shape[1])
        m = regression_metrics(preds, targets, target_names)
        valid_rmse, valid_r2 = m["rmse"], m["r2"]

        if valid_task_loss < best_valid_mae:
            best_valid_mae = valid_task_loss
            best_valid_rmse = valid_rmse
            best_valid_r2 = valid_r2
            best_metrics = dict(m, epoch=epoch)
            torch.save(model.state_dict(), best_model_path)

        per_p = " ".join(f"{n.split('_')[-1]}:{m[f'mae/{n}']:.3f}" for n in target_names)
        dev = f" | dev MAE: {m['dev_mae']:.3f} (true std {m['dev_std_true']:.3f}), dev R2: {m['dev_r2']:.3f}" if "dev_mae" in m else ""
        print(f"Epoch {epoch:03d}: Train MAE: {train_task_loss:.4f} | Valid MAE: {valid_task_loss:.4f} "
            f"[{per_p}]{dev} | LR: {scheduler.get_last_lr()[0]:.4e} |")

        row = {"epoch": epoch,
               "train_mae": train_task_loss,
               "train_aux": train_aux_loss,
               "valid_mae": valid_task_loss,
               "valid_aux": valid_aux_loss,
               "lr": scheduler.get_last_lr()[0],
               **{f"valid_{k}": v for k, v in m.items()}}
        for i in range(args.pool_num):
            row[f"valid_mod{i+1}"] = valid_mod[i].item()
            row[f"valid_k_eff{i+1}"] = valid_k_eff[i].item()
            row[f"valid_na{i+1}"] = valid_na[i]
        if args.wandb:
            wandb.log(row)

    print(f"Best model with Valid MAE: {best_valid_mae:.4f} (epoch {best_metrics.get('epoch')})")
    for n in target_names:
        print(f"  {n}: MAE {best_metrics[f'mae/{n}']:.4f}, RMSE {best_metrics[f'rmse/{n}']:.4f}, R2 {best_metrics[f'r2/{n}']:.4f}")
    if "dev_mae" in best_metrics:
        print(f"  pressure component: dev MAE {best_metrics['dev_mae']:.4f} "
            f"(true std {best_metrics['dev_std_true']:.4f}), dev R2 {best_metrics['dev_r2']:.4f}")
        print(f"  structure component: mean MAE {best_metrics['mean_mae']:.4f}")

    if args.wandb:
        wandb.log({"best_valid_mae": best_valid_mae,
                "best_valid_rmse": best_valid_rmse,
                "best_valid_r2": best_valid_r2,
                **{f"best_valid_{k}": v for k, v in best_metrics.items()}})


@torch.no_grad()
def raw_mae(Q_pred, y_heat, N_model, y_load):
    err_q = (Q_pred - y_heat).abs()
    err_n = (N_model - y_load).abs()
    return {'heat_mae': err_q.mean().item(), 'load_mae': err_n.mean().item(),
            'heat_mae_p': err_q.mean(0).cpu().numpy(), 'load_mae_p': err_n.mean(0).cpu().numpy()}


def train_pool_attn_occ(model, optimizer, scheduler, criterion, best_model_path, train_loader, valid_loader, device, args):
    if args.wandb:
        import wandb
        wandb.define_metric('epoch')
        wandb.define_metric('*', step_metric='epoch')

    heat_std = torch.tensor(args.heat_std, dtype=torch.float, device=device)
    load_std = torch.tensor(args.load_std, dtype=torch.float, device=device)
    train_m = PressureMetrics(args.pressures, heat_std, load_std)
    valid_m = PressureMetrics(args.pressures, heat_std, load_std)
    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)

    best = {'heat': float('inf'), 'epoch': None}
    heat_names = None
    load_names = [f'load_{args.gas}_{p:g}' for p in args.pressures]

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_m.reset()
        tr = {'total': 0., 'heat': 0., 'load': 0., 'aux': 0.}
        for data in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            data = data.to(device)
            optimizer.zero_grad()
            pred, extras, _, _, aux, _, _, _, _ = model(data)
            N_model = extras[5]
            assert pred.shape == data.y.shape, f'pred {tuple(pred.shape)} vs target {tuple(data.y.shape)}'

            l_heat = heat_loss(pred, data.y, heat_std)
            l_load = loading_loss(N_model, data.y_load, load_std)
            l_aux = sum(aux)
            loss = l_heat + l_load + l_aux * args.lambda_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            b = data.num_graphs
            for k, v in zip(tr, (loss, l_heat, l_load, l_aux)):
                tr[k] += v.item() * b
            train_m.update(pred, data.y, N_model, data.y_load)
        scheduler.step()
        tr = {k: v / n_train for k, v in tr.items()}

        model.eval()
        valid_m.reset()
        va = {'total': 0., 'heat': 0., 'load': 0., 'aux': 0.}
        pool = {k: [0.] * args.pool_num for k in ('mod', 'k_eff', 'na')}
        q_pred, q_true, n_pred, n_true = [], [], [], []
        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f'Epoch {epoch} [Valid]'):
                data = data.to(device)
                pred, extras, _, _, aux, s, adj, na, _ = model(data)
                N_model = extras[5]

                l_heat = heat_loss(pred, data.y, heat_std)
                l_load = loading_loss(N_model, data.y_load, load_std)
                l_aux = sum(aux)
                b = data.num_graphs
                for k, v in zip(va, (l_heat + l_load + l_aux * args.lambda_aux, l_heat, l_load, l_aux)):
                    va[k] += v.item() * b
                valid_m.update(pred, data.y, N_model, data.y_load)

                for i in range(args.pool_num):
                    pool['mod'][i] += float(soft_modularity_batch(adj[i], s[i]))
                    pool['k_eff'][i] += float(effective_num_clusters_batch(s[i]))
                    pool['na'][i] += na[i].sum().item()
                q_pred.append(pred.cpu()); q_true.append(data.y.cpu())
                n_pred.append(N_model.cpu()); n_true.append(data.y_load.cpu())
        va = {k: v / n_valid for k, v in va.items()}
        pool = {k: [x / n_valid for x in v] for k, v in pool.items()}

        q_pred, q_true = torch.cat(q_pred).numpy(), torch.cat(q_true).numpy()
        n_pred, n_true = torch.cat(n_pred).numpy(), torch.cat(n_true).numpy()
        if heat_names is None:
            heat_names = get_target_names(valid_loader, q_pred.shape[1])
        reg = {'heat': regression_metrics(q_pred, q_true, heat_names),
               'load': regression_metrics(n_pred, n_true, load_names)}

        ts, vs = train_m.summary(), valid_m.summary()
        lr = scheduler.get_last_lr()[0]

        if vs['heat'] < best['heat']:
            best = {**vs, 'epoch': epoch,
                    **{f'{k}@{p:g}': v[j].item() for k, v in valid_m.compute().items() for j, p in enumerate(args.pressures)},
                    **{f'{t}_{k}': reg[t][k] for t in reg for k in args.reg_keys if k in reg[t]},
                    **{f'{t}_r2@{p:g}': reg[t][f'r2/{n}'] for t, names in (('heat', heat_names), ('load', load_names))
                       for p, n in zip(args.pressures, names)}}
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch:03d} | train heat {ts['heat']:.3f} load {ts['load']:.4f} "
              f"| valid heat {vs['heat']:.3f} (nmae {vs['heat_nmae']:.3f}, R2 {reg['heat']['r2']:.3f}) "
              f"load {vs['load']:.4f} (nmae {vs['load_nmae']:.3f}, R2 {reg['load']['r2']:.3f}) | lr {lr:.2e}")
        print(f"          valid heat [{valid_m.line('heat')}] | load nmae [{valid_m.line('load_nmae')}] "
              f"| heat dev R2 {reg['heat'].get('dev_r2', float('nan')):.3f}")

        if args.wandb:
            row = {'epoch': epoch, 'train/lr': lr,
                   **valid_m.wandb_sections(args.gas),
                   **{f'train/loss_{k}': v for k, v in tr.items()},
                   **{f'train/{k}': v for k, v in ts.items()},
                   **{f'valid/loss_{k}': v for k, v in va.items()},
                   **{f'valid/{t}_{k}': reg[t][k] for t in reg for k in args.reg_keys if k in reg[t]}}
            for i in range(args.pool_num):
                for k, v in pool.items():
                    row[f'pool/{k}{i + 1}'] = v[i]
            wandb.log(row)

    print(f"Best model: epoch {best['epoch']}")
    print(f"  heat: MAE {best['heat']:.4f} (nmae {best['heat_nmae']:.4f}) | RMSE {best['heat_rmse']:.4f} | "
          f"R2 {best['heat_r2']:.4f} | dev R2 {best.get('heat_dev_r2', float('nan')):.4f} "
          f"(true dev std {reg['heat'].get('dev_std_true', float('nan')):.3f}) | mean MAE {best.get('heat_mean_mae', float('nan')):.4f}")
    print(f"  load: MAE {best['load']:.4f} (nmae {best['load_nmae']:.4f}) | RMSE {best['load_rmse']:.4f} | "
          f"R2 {best['load_r2']:.4f} | dev R2 {best.get('load_dev_r2', float('nan')):.4f} "
          f"(true dev std {reg['load'].get('dev_std_true', float('nan')):.3f}) | mean MAE {best.get('load_mean_mae', float('nan')):.4f}")
    for p in args.pressures:
        print(f"  {p:g} bar: heat {best[f'heat@{p:g}']:.4f} (nmae {best[f'heat_nmae@{p:g}']:.4f}, R2 {best[f'heat_r2@{p:g}']:.4f}) | "
              f"load {best[f'load@{p:g}']:.4f} (nmae {best[f'load_nmae@{p:g}']:.4f}, R2 {best[f'load_r2@{p:g}']:.4f})")
    if args.wandb:
        wandb.run.summary.update({f'best/{k}': v for k, v in best.items()})
