import os
import csv
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from .utils import soft_modularity_batch, effective_num_clusters_batch, PressureMetrics, HeatMetrics


def dict2csv(model_path, result, args):
    csv_path = Path(model_path)
    csv_path = csv_path.parent / "result.csv"
    file_exists = os.path.exists(csv_path)

    fieldnames = ["seed", "data_path", "target"]
    fieldnames.extend(result.keys())
    result['seed'] = args.seed
    result['data_path'] = args.data_path
    result['target'] = args.target

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(result)


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


def test_base(model, optimizer, criterion, best_model_path, test_loader, device, args):
    if args.wandb:
        import wandb

    test_length = len(test_loader.dataset)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_task_loss = 0.0
    preds, targets = [], []
    target_names = None

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            pred = model(data)

            target = data.y
            task_loss = criterion(pred, target)

            test_task_loss += task_loss.item() * data.num_graphs
            preds.append(pred.cpu())
            targets.append(target.cpu())

    test_task_loss /= test_length
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    if target_names is None:
        target_names = get_target_names(test_loader, preds.shape[1])
    m = regression_metrics(preds, targets, target_names)
    test_rmse, test_r2 = m["rmse"], m["r2"]


    print(f"Best model with Test MAE: {test_task_loss:.4f}")
    for n in target_names:
        print(f"  {n}: MAE {m[f'mae/{n}']:.4f}, RMSE {m[f'rmse/{n}']:.4f}, R2 {m[f'r2/{n}']:.4f}")
    if "dev_mae" in m:
        print(f"  pressure component: dev MAE {m['dev_mae']:.4f} "
            f"(true std {m['dev_std_true']:.4f}), dev R2 {m['dev_r2']:.4f}")
        print(f"  structure component: mean MAE {m['mean_mae']:.4f}")

    log_dict = {"test_mae": test_task_loss,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                **{f"best_test_{k}": v for k, v in m.items()}}
    if args.wandb:
        wandb.log(log_dict)

    dict2csv(best_model_path, log_dict, args)


def test_base_heat(model, optimizer, criterion, best_model_path, test_loader, device, args):
    if args.wandb:
        import wandb

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    heat_std = torch.tensor(args.heat_std, dtype=torch.float, device=device)
    test_m = HeatMetrics(args.pressures, heat_std)
    n_test = len(test_loader.dataset)

    te = {'total': 0., 'heat': 0.}
    q_pred, q_true = [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Test'):
            data = data.to(device)
            pred = model(data)
            assert pred.shape == data.y.shape, f'pred {tuple(pred.shape)} vs target {tuple(data.y.shape)}'

            l_heat = heat_loss(pred, data.y, heat_std)
            b = data.num_graphs
            for k, v in zip(te, (l_heat, l_heat)):
                te[k] += v.item() * b
            test_m.update(pred, data.y)

            q_pred.append(pred.cpu()); q_true.append(data.y.cpu())
    te = {k: v / n_test for k, v in te.items()}

    q_pred, q_true = torch.cat(q_pred).numpy(), torch.cat(q_true).numpy()
    heat_names = get_target_names(test_loader, q_pred.shape[1])
    reg = {'heat': regression_metrics(q_pred, q_true, heat_names)}

    res = {**test_m.summary(),
           **{f'{k}@{p:g}': v[j].item() for k, v in test_m.compute().items() for j, p in enumerate(args.pressures)},
           **{f'heat_{k}': reg['heat'][k] for k in args.reg_keys if k in reg['heat']},
           **{f'heat_r2@{p:g}': reg['heat'][f'r2/{n}'] for p, n in zip(args.pressures, heat_names)},
           **{f'loss_{k}': v for k, v in te.items()}}
    log_dict = {f'test_{k}': v for k, v in res.items()}

    print('Test (best checkpoint)')
    print(f"  heat: MAE {res['heat']:.4f} (nmae {res['heat_nmae']:.4f}) | RMSE {res['heat_rmse']:.4f} | "
          f"R2 {res['heat_r2']:.4f} | dev R2 {res.get('heat_dev_r2', float('nan')):.4f} "
          f"(true dev std {res.get('heat_dev_std_true', float('nan')):.3f}) | mean MAE {res.get('heat_mean_mae', float('nan')):.4f}")
    for p in args.pressures:
        print(f"  {p:g} bar: heat {res[f'heat@{p:g}']:.4f} (nmae {res[f'heat_nmae@{p:g}']:.4f}, R2 {res[f'heat_r2@{p:g}']:.4f})")

    if args.wandb:
        wandb.run.summary.update({f'test/{k}': v for k, v in res.items()})

    dict2csv(best_model_path, log_dict, args)


def test_base_heat_load(model, optimizer, criterion, best_model_path, test_loader, device, args):
    if args.wandb:
        import wandb

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    heat_std = torch.tensor(args.heat_std, dtype=torch.float, device=device)
    load_std = torch.tensor(args.load_std, dtype=torch.float, device=device)
    test_m = PressureMetrics(args.pressures, heat_std, load_std)
    n_test = len(test_loader.dataset)
    load_names = [f'load_{args.gas}_{p:g}' for p in args.pressures]

    te = {'total': 0., 'heat': 0., 'load': 0.}
    q_pred, q_true, n_pred, n_true = [], [], [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Test'):
            data = data.to(device)
            pred, N_model = model(data)
            assert pred.shape == data.y.shape, f'pred {tuple(pred.shape)} vs target {tuple(data.y.shape)}'
            assert N_model.shape == data.y_load.shape, f'load {tuple(N_model.shape)} vs target {tuple(data.y_load.shape)}'

            l_heat = heat_loss(pred, data.y, heat_std)
            l_load = loading_loss(N_model, data.y_load, load_std)
            b = data.num_graphs
            for k, v in zip(te, (l_heat + l_load, l_heat, l_load)):
                te[k] += v.item() * b
            test_m.update(pred, data.y, N_model, data.y_load)

            q_pred.append(pred.cpu()); q_true.append(data.y.cpu())
            n_pred.append(N_model.cpu()); n_true.append(data.y_load.cpu())
    te = {k: v / n_test for k, v in te.items()}

    q_pred, q_true = torch.cat(q_pred).numpy(), torch.cat(q_true).numpy()
    n_pred, n_true = torch.cat(n_pred).numpy(), torch.cat(n_true).numpy()
    heat_names = get_target_names(test_loader, q_pred.shape[1])
    reg = {'heat': regression_metrics(q_pred, q_true, heat_names),
           'load': regression_metrics(n_pred, n_true, load_names)}

    res = {**test_m.summary(),
           **{f'{k}@{p:g}': v[j].item() for k, v in test_m.compute().items() for j, p in enumerate(args.pressures)},
           **{f'{t}_{k}': reg[t][k] for t in reg for k in args.reg_keys if k in reg[t]},
           **{f'{t}_r2@{p:g}': reg[t][f'r2/{n}'] for t, names in (('heat', heat_names), ('load', load_names))
              for p, n in zip(args.pressures, names)},
           **{f'loss_{k}': v for k, v in te.items()}}
    log_dict = {f'test_{k}': v for k, v in res.items()}

    print('Test (best checkpoint)')
    print(f"  heat: MAE {res['heat']:.4f} (nmae {res['heat_nmae']:.4f}) | RMSE {res['heat_rmse']:.4f} | "
          f"R2 {res['heat_r2']:.4f} | dev R2 {res.get('heat_dev_r2', float('nan')):.4f} "
          f"(true dev std {res.get('heat_dev_std_true', float('nan')):.3f}) | mean MAE {res.get('heat_mean_mae', float('nan')):.4f}")
    print(f"  load: MAE {res['load']:.4f} (nmae {res['load_nmae']:.4f}) | RMSE {res['load_rmse']:.4f} | "
          f"R2 {res['load_r2']:.4f} | dev R2 {res.get('load_dev_r2', float('nan')):.4f} "
          f"(true dev std {res.get('load_dev_std_true', float('nan')):.3f}) | mean MAE {res.get('load_mean_mae', float('nan')):.4f}")
    for p in args.pressures:
        print(f"  {p:g} bar: heat {res[f'heat@{p:g}']:.4f} (nmae {res[f'heat_nmae@{p:g}']:.4f}, R2 {res[f'heat_r2@{p:g}']:.4f}) | "
              f"load {res[f'load@{p:g}']:.4f} (nmae {res[f'load_nmae@{p:g}']:.4f}, R2 {res[f'load_r2@{p:g}']:.4f})")

    if args.wandb:
        wandb.run.summary.update({f'test/{k}': v for k, v in res.items()})

    dict2csv(best_model_path, log_dict, args)


def test_pool(model, optimizer, criterion, best_model_path, test_loader, device, args):
    if args.wandb:
        import wandb

    test_length = len(test_loader.dataset)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_total_loss = 0.0
    test_task_loss = 0.0
    test_aux_loss = 0.0
    test_mod = [0. for _ in range(args.pool_num)]
    test_k_eff = [0. for _ in range(args.pool_num)]
    test_na = [0. for _ in range(args.pool_num)]
    preds, targets = [], []
    target_names = None

    with torch.no_grad():
        for data in tqdm(test_loader):
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
            assert pred.shape == target.shape, f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            task_loss = criterion(pred, target)
            aux_loss = sum(aux)
            total_loss = task_loss + aux_loss

            test_total_loss += total_loss.item() * data.num_graphs
            test_task_loss += task_loss.item() * data.num_graphs
            test_aux_loss += aux_loss.item() * data.num_graphs
            for i in range(args.pool_num):
                test_mod[i] += soft_modularity_batch(adj[i], s[i])
                test_k_eff[i] += effective_num_clusters_batch(s[i])
                test_na[i] += na[i].sum().item()
            preds.append(pred.cpu())
            targets.append(target.cpu())

    test_total_loss /= test_length
    test_task_loss /= test_length
    test_aux_loss /= test_length
    for i in range(args.pool_num):
        test_mod[i] /= test_length
        test_k_eff[i] /= test_length
        test_na[i] /= test_length
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    if target_names is None:
        target_names = get_target_names(test_loader, preds.shape[1])
    m = regression_metrics(preds, targets, target_names)
    test_rmse, test_r2 = m["rmse"], m["r2"]

    print(f"Best model with Test MAE: {test_task_loss:.4f}")
    for n in target_names:
        print(f"  {n}: MAE {m[f'mae/{n}']:.4f}, RMSE {m[f'rmse/{n}']:.4f}, R2 {m[f'r2/{n}']:.4f}")
    if "dev_mae" in m:
        print(f"  pressure component: dev MAE {m['dev_mae']:.4f} "
            f"(true std {m['dev_std_true']:.4f}), dev R2 {m['dev_r2']:.4f}")
        print(f"  structure component: mean MAE {m['mean_mae']:.4f}")

    log_dict = {"test_mae": test_task_loss,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                **{f"best_test_{k}": v for k, v in m.items()}}
    for i in range(args.pool_num):
        log_dict[f"test_mod{i+1}"] = test_mod[i].item()
        log_dict[f"test_k_eff{i+1}"] = test_k_eff[i].item()
        log_dict[f"test_na{i+1}"] = test_na[i]
    if args.wandb:
        wandb.log(log_dict)

    dict2csv(best_model_path, log_dict, args)


def test_pool_attn(model, optimizer, criterion, best_model_path, test_loader, device, args):
    if args.wandb:
        import wandb

    test_length = len(test_loader.dataset)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_total_loss = 0.0
    test_task_loss = 0.0
    test_aux_loss = 0.0
    test_mod = [0. for _ in range(args.pool_num)]
    test_k_eff = [0. for _ in range(args.pool_num)]
    test_na = [0. for _ in range(args.pool_num)]
    preds, targets = [], []
    target_names = None

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            pred, e, m, alpha, aux, s, adj, na, _ = model(data)

            target = data.y
            assert pred.shape == target.shape, f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            task_loss = criterion(pred, target)
            aux_loss = sum(aux)
            total_loss = task_loss + aux_loss

            test_total_loss += total_loss.item() * data.num_graphs
            test_task_loss += task_loss.item() * data.num_graphs
            test_aux_loss += aux_loss.item() * data.num_graphs
            for i in range(args.pool_num):
                test_mod[i] += soft_modularity_batch(adj[i], s[i])
                test_k_eff[i] += effective_num_clusters_batch(s[i])
                test_na[i] += na[i].sum().item()
            preds.append(pred.cpu())
            targets.append(target.cpu())

    test_total_loss /= test_length
    test_task_loss /= test_length
    test_aux_loss /= test_length
    for i in range(args.pool_num):
        test_mod[i] /= test_length
        test_k_eff[i] /= test_length
        test_na[i] /= test_length
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    if target_names is None:
        target_names = get_target_names(test_loader, preds.shape[1])
    m = regression_metrics(preds, targets, target_names)
    test_rmse, test_r2 = m["rmse"], m["r2"]

    print(f"Best model with Test MAE: {test_task_loss:.4f}")
    for n in target_names:
        print(f"  {n}: MAE {m[f'mae/{n}']:.4f}, RMSE {m[f'rmse/{n}']:.4f}, R2 {m[f'r2/{n}']:.4f}")
    if "dev_mae" in m:
        print(f"  pressure component: dev MAE {m['dev_mae']:.4f} "
            f"(true std {m['dev_std_true']:.4f}), dev R2 {m['dev_r2']:.4f}")
        print(f"  structure component: mean MAE {m['mean_mae']:.4f}")

    log_dict = {"test_mae": test_task_loss,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                **{f"best_test_{k}": v for k, v in m.items()}}
    for i in range(args.pool_num):
        log_dict[f"test_mod{i+1}"] = test_mod[i].item()
        log_dict[f"test_k_eff{i+1}"] = test_k_eff[i].item()
        log_dict[f"test_na{i+1}"] = test_na[i]
    if args.wandb:
        wandb.log(log_dict)

    dict2csv(best_model_path, log_dict, args)


@torch.no_grad()
def raw_mae(Q_pred, y_heat, N_model, y_load):
    err_q = (Q_pred - y_heat).abs()
    err_n = (N_model - y_load).abs()
    return {'heat_mae': err_q.mean().item(), 'load_mae': err_n.mean().item(),
            'heat_mae_p': err_q.mean(0).cpu().numpy(), 'load_mae_p': err_n.mean(0).cpu().numpy()}


def test_pool_attn_occ(model, optimizer, criterion, best_model_path, test_loader, device, save, args):
    if args.wandb:
        import wandb

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    heat_std = torch.tensor(args.heat_std, dtype=torch.float, device=device)
    load_std = torch.tensor(args.load_std, dtype=torch.float, device=device)
    test_m = PressureMetrics(args.pressures, heat_std, load_std)
    n_test = len(test_loader.dataset)
    load_names = [f'load_{args.gas}_{p:g}' for p in args.pressures]

    te = {'total': 0., 'heat': 0., 'load': 0., 'aux': 0.}
    pool = {k: [0.] * args.pool_num for k in ('mod', 'k_eff', 'na')}
    q_pred, q_true, n_pred, n_true = [], [], [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Test'):
            data = data.to(device)
            pred, extras, _, _, aux, s, adj, na, _ = model(data)
            N_model = extras[5]
            assert pred.shape == data.y.shape, f'pred {tuple(pred.shape)} vs target {tuple(data.y.shape)}'

            l_heat = heat_loss(pred, data.y, heat_std)
            l_load = loading_loss(N_model, data.y_load, load_std)
            l_aux = sum(aux)
            b = data.num_graphs
            for k, v in zip(te, (l_heat + l_load + l_aux * args.lambda_aux, l_heat, l_load, l_aux)):
                te[k] += v.item() * b
            test_m.update(pred, data.y, N_model, data.y_load)

            for i in range(args.pool_num):
                pool['mod'][i] += float(soft_modularity_batch(adj[i], s[i]))
                pool['k_eff'][i] += float(effective_num_clusters_batch(s[i]))
                pool['na'][i] += na[i].sum().item()
            q_pred.append(pred.cpu()); q_true.append(data.y.cpu())
            n_pred.append(N_model.cpu()); n_true.append(data.y_load.cpu())
    te = {k: v / n_test for k, v in te.items()}
    pool = {k: [x / n_test for x in v] for k, v in pool.items()}

    q_pred, q_true = torch.cat(q_pred).numpy(), torch.cat(q_true).numpy()
    n_pred, n_true = torch.cat(n_pred).numpy(), torch.cat(n_true).numpy()
    heat_names = get_target_names(test_loader, q_pred.shape[1])
    reg = {'heat': regression_metrics(q_pred, q_true, heat_names),
           'load': regression_metrics(n_pred, n_true, load_names)}

    res = {**test_m.summary(),
           **{f'{k}@{p:g}': v[j].item() for k, v in test_m.compute().items() for j, p in enumerate(args.pressures)},
           **{f'{t}_{k}': reg[t][k] for t in reg for k in args.reg_keys if k in reg[t]},
           **{f'{t}_r2@{p:g}': reg[t][f'r2/{n}'] for t, names in (('heat', heat_names), ('load', load_names))
              for p, n in zip(args.pressures, names)},
           **{f'loss_{k}': v for k, v in te.items()}}
    for i in range(args.pool_num):
        for k, v in pool.items():
            res[f'{k}{i + 1}'] = v[i]
    log_dict = {f'test_{k}': v for k, v in res.items()}

    print('Test (best checkpoint)')
    print(f"  heat: MAE {res['heat']:.4f} (nmae {res['heat_nmae']:.4f}) | RMSE {res['heat_rmse']:.4f} | "
          f"R2 {res['heat_r2']:.4f} | dev R2 {res.get('heat_dev_r2', float('nan')):.4f} "
          f"(true dev std {res.get('heat_dev_std_true', float('nan')):.3f}) | mean MAE {res.get('heat_mean_mae', float('nan')):.4f}")
    print(f"  load: MAE {res['load']:.4f} (nmae {res['load_nmae']:.4f}) | RMSE {res['load_rmse']:.4f} | "
          f"R2 {res['load_r2']:.4f} | dev R2 {res.get('load_dev_r2', float('nan')):.4f} "
          f"(true dev std {res.get('load_dev_std_true', float('nan')):.3f}) | mean MAE {res.get('load_mean_mae', float('nan')):.4f}")
    for p in args.pressures:
        print(f"  {p:g} bar: heat {res[f'heat@{p:g}']:.4f} (nmae {res[f'heat_nmae@{p:g}']:.4f}, R2 {res[f'heat_r2@{p:g}']:.4f}) | "
              f"load {res[f'load@{p:g}']:.4f} (nmae {res[f'load_nmae@{p:g}']:.4f}, R2 {res[f'load_r2@{p:g}']:.4f})")

    if args.wandb:
        wandb.run.summary.update({f'test/{k}': v for k, v in res.items()})
    if save:
        dict2csv(best_model_path, log_dict, args)
