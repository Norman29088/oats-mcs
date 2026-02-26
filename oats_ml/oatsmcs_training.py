#==================================================================
# oatsmcs_training.py
# OATS-MCS Neural Network Demo Training (Preventive DC-SCOPF)
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
# Last updated: 2026-02
#==================================================================


# from typing import Any, Dict, List, Optional
import sqlite3
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from oats_val.oatsmcs_validation import (
    OATSMCS_MAP,
    read_testcase_assets,
    clean_and_dim_check,
    to_torch_blocks,
    load_grid_params,
    power_flow_sc,
    fmax_violation,
    check_consistency_residual_analysis,
    get_pg_bounds,gen_bound_violation_analysis,pw_bound_violation_analysis
)

# Model
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, n_hidden: int = 2, use_bn=False):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def read_dataset_sql_by_scen(db_path, scen_list):
    data = {}
    scen_str = ",".join(map(str, scen_list))
    with sqlite3.connect(db_path) as conn:
        for key, table_name in OATSMCS_MAP.items():
            query = f'''
                SELECT * FROM "{table_name}"
                WHERE scen IN ({scen_str})
                ORDER BY scen
            '''
            data[key] = pd.read_sql_query(query, conn)
    return data




def clip_by_bounds(y_pg, y_pw, pg_lb, pg_ub, pw_lb, pw_ub):
    y_pg = torch.clamp(y_pg, min=pg_lb, max=pg_ub)
    y_pw = torch.clamp(y_pw, min=pw_lb, max=pw_ub)
    return y_pg, y_pw



def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sc_power_flow_analysis(data: dict, pred_pg: torch.Tensor, pred_pw: torch.Tensor,
                           PTDF_SC: torch.Tensor, G2B: torch.Tensor, W2B: torch.Tensor, D2B: torch.Tensor,
                           Fmax_SC: torch.Tensor, atol: float = 1e-6, rtol: float = 0.0,
                           vio_name: str = "sc_fmax_violation", verbose: bool = False) -> dict:
    data_pred = {"PD": data["PD"], "PW": pred_pw, "pG": pred_pg}
    PF_sc_pred = power_flow_sc(data_pred, PTDF_SC, G2B, W2B, D2B)
    viol_sc = fmax_violation(PF_sc_pred, Fmax_SC, atol=atol, rtol=rtol)

    report = check_consistency_residual_analysis(viol_sc, torch.zeros_like(viol_sc), atol=0.0, rtol=0.0, name=vio_name, verbose=verbose)
    return report,viol_sc.pow(2).mean()



def reduce_reports(reports: list[dict]) -> dict:
    if not reports:
        return {"violation_count": 0, "maximum_absolute_difference": 0.0,
                "mean_absolute_difference": 0.0, "percentile95_absolute_difference": 0.0,
                "total_number_of_elements": 0}

    first = reports[0]
    out = {"analysis_name": first.get("analysis_name"),
           "absolute_tolerance": first.get("absolute_tolerance"),
           "relative_tolerance": first.get("relative_tolerance"),
           "tensor_shape": first.get("tensor_shape")}

    viol_cnt_sum = 0; max_abs = 0.0; p95_abs = 0.0; numel_sum = 0; mean_abs_num = 0.0

    for r in reports:
        viol_cnt_sum += int(r.get("violation_count", 0))
        max_abs = max(max_abs, float(r.get("maximum_absolute_difference", 0.0)))
        p95_abs = max(p95_abs, float(r.get("percentile95_absolute_difference", 0.0)))
        ne = int(r.get("total_number_of_elements", 0))
        ma = float(r.get("mean_absolute_difference", 0.0))
        numel_sum += ne; mean_abs_num += ma * ne

    out.update({"violation_count": int(viol_cnt_sum),
                "maximum_absolute_difference": float(max_abs),
                "mean_absolute_difference": float(mean_abs_num / numel_sum) if numel_sum > 0 else 0.0,
                "percentile95_absolute_difference": float(p95_abs),
                "total_number_of_elements": int(numel_sum),
                "n_batches": int(len(reports))})
    return out


def dispatch_analysis(
    oats_dict: dict,
    *,
    pred_pg: torch.Tensor,
    pred_pw: torch.Tensor,
    true_pg: torch.Tensor,
    true_pw: torch.Tensor,
    PD: torch.Tensor,
    PW_ub: torch.Tensor,
    split: str,
    verbose: bool = False,
) -> dict:
    """
    Return scalar metrics (epoch/batch aggregatable) for:
      - cost gap: C(pred) - C(true)
      - power balance residual: |sum(pg)+sum(pw)-sum(pd)|
    Shapes:
      pred_pg/true_pg: (B, nG)
      pred_pw/true_pw: (B, nW)
      PD:             (B, nD)
      PW_ub:          (B, nW) or (1, nW)
    """
    device = pred_pg.device
    dtype = pred_pg.dtype

    c2 = torch.as_tensor(oats_dict["generator"]["costc2"], device=device, dtype=dtype).view(1, -1)
    c1 = torch.as_tensor(oats_dict["generator"]["costc1"], device=device, dtype=dtype).view(1, -1)
    c0 = torch.as_tensor(oats_dict["generator"]["costc0"], device=device, dtype=dtype).view(1, -1)
    offer = torch.as_tensor(oats_dict["renewable"]["offer"], device=device, dtype=dtype).view(1, -1)

    # --- generator cost ---
    cost_g_pred = (c2 * pred_pg * pred_pg + c1 * pred_pg + c0).sum(dim=1)
    cost_g_true = (c2 * true_pg * true_pg + c1 * true_pg + c0).sum(dim=1)

    # --- renewable curtailment "cost": offer*(ub - pw) ---
    cost_w_pred = (offer * (PW_ub - pred_pw)).sum(dim=1)
    cost_w_true = (offer * (PW_ub - true_pw)).sum(dim=1)

    cost_pred = cost_g_pred + cost_w_pred
    cost_true = cost_g_true + cost_w_true
    eps=0
    cost_gap = (cost_pred - cost_true) / (cost_true.abs() + eps)

    # power balance percentage residual (per-scenario) -
    pd_sum = PD.sum(dim=1)                                   # (B,)
    inj_res = pred_pg.sum(dim=1) + pred_pw.sum(dim=1) - pd_sum
    pb = torch.abs(inj_res) / pd_sum.clamp_min(1e-8)         # (B,)
    
    rep = {
        "split": split,
        "n_samples": int(pred_pg.shape[0]),

        "cost_gap_mean": cost_gap.mean().item(),
        "cost_gap_median": cost_gap.median().item(),
        "cost_gap_max_abs": cost_gap.abs().max().item(),

        "pb_mean": pb.mean().item(),
        "pb_median": pb.median().item(),
        "pb_max": pb.max().item(),
    }

    if verbose:
        print(f"[dispatch] {split} n={rep['n_samples']} cost_gap_mean={rep['cost_gap_mean']:.6g} pb_mean={rep['pb_mean']:.6g}")

    return rep

def reduce_dispatch(reps: list[dict]) -> dict:
    if not reps:
        return {}
    keys = ["cost_gap_mean", "cost_gap_median", "cost_gap_max_abs", "pb_mean", "pb_median", "pb_max"]
    out = {"split": reps[0].get("split", ""), "n_batches": len(reps), "n_samples": int(sum(r.get("n_samples", 0) for r in reps))}
    for k in keys:
        v = np.array([r[k] for r in reps if k in r], dtype=float)
        if v.size == 0:
            continue
        out[k] = float(np.mean(v))
    return out

def eval_loss_on_scens(model, oatsmcs_data,  oats_dict, pg_lb, pg_ub, batch_size, device, dtype,
                    PTDF_SC, G2B, W2B, D2B, Fmax_SC, *, split: str, epoch: int, atol: float, rtol: float,use_clip=True,):
    model.eval()
    loss_sum = 0.0
    n_sum = 0
    sc_report_all, gen_report_all, pw_report_all, disp_report_all = [], [], [], []
    vio_name = f"{split}_flow_violation"

    N = int(oatsmcs_data["PD"].shape[0])
    n_batch = (N + batch_size - 1) // batch_size

    with torch.no_grad():
        for j in range(n_batch):
            start = j * batch_size
            end = min((j + 1) * batch_size, N)
            if start >= end:
                break

            PD_b = oatsmcs_data["PD"][start:end]
            PW_b = oatsmcs_data["PW"][start:end]
            Y_pg = oatsmcs_data["pG"][start:end]
            Y_pw = oatsmcs_data["PW"][start:end]
            PW_lb = oatsmcs_data["PW_LB"][start:end]
            PW_ub = oatsmcs_data["PW_UB"][start:end]

            X = torch.cat([PD_b, PW_b], dim=1)

            pred = model(X)
            nG = Y_pg.shape[1]
            pred_pg, pred_pw = pred[:, :nG], pred[:, nG:]
            if use_clip:
                pred_pg, pred_pw = clip_by_bounds(pred_pg, pred_pw, pg_lb, pg_ub, PW_lb, PW_ub)

            data_batch = {"PD": PD_b, "PW": pred_pw, "pG": pred_pg}
            sc_vio_rep, pf_penalty_val = sc_power_flow_analysis(data_batch, pred_pg, pred_pw, PTDF_SC, G2B, W2B, D2B, Fmax_SC, atol, 
                                                                rtol, vio_name, False)
            sc_report_all.append(sc_vio_rep)

            loss = F.mse_loss(pred_pg, Y_pg) + F.mse_loss(pred_pw, Y_pw)+ 0.9 * pf_penalty_val
            loss_sum += float(loss.item()) * X.shape[0]
            n_sum += X.shape[0]

            gen_vio_rep = gen_bound_violation_analysis(pred_pg, pg_lb, pg_ub, split=split, verbose=False)
            gen_report_all.append(gen_vio_rep)
            pw_vio_rep = pw_bound_violation_analysis(pred_pw, PW_lb, PW_ub, split=split, verbose=False)
            pw_report_all.append(pw_vio_rep)
            rep_disp = dispatch_analysis(oats_dict, pred_pg=pred_pg, pred_pw=pred_pw, true_pg=Y_pg, true_pw=Y_pw,
                                        PD=data_batch["PD"], PW_ub=PW_ub, split=split, verbose=False)
            disp_report_all.append(rep_disp)
            del data_batch, X, Y_pg, Y_pw, pred, pred_pg, pred_pw

    sc_epoch_report = reduce_reports(sc_report_all)
    sc_epoch_report.update({"split": split, "epoch": int(epoch), "n_samples": int(n_sum), "n_batches": int(len(sc_report_all))})
    sc_report_all.clear()

    gen_epoch_report = reduce_reports(gen_report_all)
    gen_epoch_report.update({"split": split, "epoch": int(epoch), "n_samples": int(n_sum), "n_batches": int(len(gen_report_all))})
    gen_report_all.clear()

    pw_epoch_report = reduce_reports(pw_report_all)
    pw_epoch_report.update({"split": split, "epoch": int(epoch), "n_samples": int(n_sum), "n_batches": int(len(pw_report_all))})
    pw_report_all.clear()

    disp_epoch_report = reduce_dispatch(disp_report_all)
    disp_epoch_report.update({"split": split, "epoch": int(epoch), "n_samples": int(n_sum), "n_batches": int(len(disp_report_all))})
    disp_report_all.clear()

    model.train()
    return loss_sum / max(n_sum, 1), sc_epoch_report, gen_epoch_report, pw_epoch_report, disp_epoch_report


def train_scopf_mlp(
    testcase="IEEE-24R/IEEE-24R.xlsx",
    resultcase="IEEE-24R/results.xlsx",
    db_path="IEEE-24R/IEEE24_demo_30r.db",
    grid_param_path="IEEE-24R/N-1_contingency_test",
    batch_size=None,
    split_ratio=(8, 1, 1),
    atol=1e-5,
    rtol=1e-5,
    device="cpu",
    dtype=torch.float64,
    random_seed=33000,
    hidden_dim=64,
    n_hidden=2,
    epochs=10,
    lr=1e-3,
    use_clip=True,
    use_bn=True
):
    
    # Step 1: fixed seed
    set_random_seed(int(random_seed))
    device = torch.device(device)

    # Step 2: load grid params 
    oats_dict = read_testcase_assets(testcase)
    PTDF_SC_full, Fmax_SC_full,G2B, W2B, D2B= load_grid_params(grid_param_path, use_sc=True, device=device, dtype=dtype)
    pg_lb, pg_ub = get_pg_bounds(oats_dict, device=device, dtype=dtype)  # (nG,), (nG,)

    # Step 3: Initialize model
    model = MLP(D2B.shape[1]+W2B.shape[1], G2B.shape[1]+W2B.shape[1], hidden_dim=hidden_dim, n_hidden=n_hidden,
                use_bn=use_bn).to(device=device, dtype=dtype)
    model = model.to(device)

    # Step 4: Get split ratio
    train_ratio, val_ratio, test_ratio = split_ratio
    total_ratio = train_ratio + val_ratio + test_ratio
    frac_train = train_ratio / total_ratio
    frac_val = val_ratio / total_ratio

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    # Get global permutation
    with sqlite3.connect(db_path) as conn:
        scen_all = pd.read_sql_query('SELECT scen FROM "Generator_Real" ORDER BY scen', conn)["scen"].to_numpy()

    total_scen = int(len(scen_all))

    if batch_size is None:
        batch_size =total_scen # min(total_scen, 4096)-TEMP
    else:
        batch_size = max(1, min(int(batch_size), total_scen))
    print(f"[MLP] Batch size: {batch_size}")

    perm = np.random.RandomState(random_seed).permutation(total_scen)
    scen_all = scen_all[perm]

    n_train = int(total_scen * frac_train)
    n_val   = int(total_scen * frac_val)

    scen_train_all = scen_all[:n_train]
    scen_val_all   = scen_all[n_train:n_train+n_val]
    scen_test_all  = scen_all[n_train+n_val:]

    def build_oatsmcs_data(db_path,scen_train_all,oats_dict, device, dtype):
        data_train = read_dataset_sql_by_scen(db_path, scen_train_all)
        data_train = clean_and_dim_check(data_train, oats_dict)
        data_train = to_torch_blocks(
            data_train,
            keys=["pG","PD","PW","PF","PW_UB","PW_LB"],
            device=device,
            dtype=dtype
        )
        return data_train

    print(f"Split train data... n={len(scen_train_all)}")
    data_train = build_oatsmcs_data(db_path, scen_train_all, oats_dict, device=device, dtype=dtype)
    print(f"Split val data...   n={len(scen_val_all)}")
    data_val = build_oatsmcs_data(db_path, scen_val_all, oats_dict, device=device, dtype=dtype)
    print(f"Split test data...  n={len(scen_test_all)}")
    data_test = build_oatsmcs_data(db_path, scen_test_all, oats_dict, device=device, dtype=dtype)

    X_train_full = torch.cat([data_train["PD"], data_train["PW"]], dim=1)
    Ypg_train_full = data_train["pG"]
    Ypw_train_full = data_train["PW"]
    PW_lb_full = data_train["PW_LB"]
    PW_ub_full = data_train["PW_UB"]

    n_batch_train  = (len(scen_train_all) + batch_size - 1) // batch_size

    train_loss_hist, val_loss_hist = [], []
    test_loss_hist = []
    train_vio_hist, val_vio_hist = [], []
    test_vio = None
    train_gen_hist, val_gen_hist = [], []
    train_pw_hist,  val_pw_hist  = [], []
    train_dispatch_hist, val_dispatch_hist = [], []
    test_gen = None
    test_pw = None
    test_dispatch = None

    for epoch in range(int(epochs)):
        batch_idx = 0
        loss_train_sum = 0.0
        n_train_sum = 0
        sc_report_all_train, gen_report_all_train, pw_report_all_train, disp_report_all_train = [], [], [], []

        pbar = tqdm(total=n_batch_train, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        for batch_idx in range(n_batch_train):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(scen_train_all))
            if (end - start) <= 1:
                pbar.update(1)
                continue
            if start >= end: break

            X = X_train_full[start:end]
            Y_pg = Ypg_train_full[start:end]
            Y_pw = Ypw_train_full[start:end]
            PW_lb = PW_lb_full[start:end]
            PW_ub = PW_ub_full[start:end]

            X_train = X; Ypg_train = Y_pg; Ypw_train = Y_pw

            pred_train = model(X_train)
            nG = Ypg_train.shape[1]
            pred_pg = pred_train[:, :nG]; pred_pw = pred_train[:, nG:]
            # if use_clip: 
            #     pred_pg, pred_pw = clip_by_bounds(pred_pg, pred_pw, pg_lb, pg_ub, PW_lb, PW_ub)

            data = {
                "PD": data_train["PD"][start:end],
                "PW": pred_pw,
                "pG": pred_pg,
            }
            sc_vio_rep, pf_penalty = sc_power_flow_analysis(data, pred_pg, pred_pw, PTDF_SC_full, G2B, W2B, D2B, Fmax_SC_full, atol, 
                                                            rtol, "train_flow_violation", False)
            sc_report_all_train.append(sc_vio_rep)

            loss = F.mse_loss(pred_pg, Y_pg) + F.mse_loss(pred_pw, Y_pw) + 0.9 * pf_penalty
            loss_train_sum += float(loss.item()) * X.shape[0]

            gen_vio_rep = gen_bound_violation_analysis(pred_pg, pg_lb, pg_ub, split="train", verbose=False)
            gen_report_all_train.append(gen_vio_rep)
            pw_vio_rep = pw_bound_violation_analysis(pred_pw, PW_lb, PW_ub, split="train", verbose=False)
            pw_report_all_train.append(pw_vio_rep)

            rep_disp = dispatch_analysis(oats_dict, pred_pg=pred_pg, pred_pw=pred_pw, true_pg=Y_pg, true_pw=Y_pw,
                                        PD=data["PD"], PW_ub=PW_ub, split="train", verbose=False)
            disp_report_all_train.append(rep_disp)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            n_train_sum += X_train.shape[0]
            pbar.set_postfix(train_loss=loss_train_sum/max(n_train_sum,1))
            
            del data, X, Y_pg, Y_pw, X_train, pred_train, pred_pg, pred_pw
        pbar.close()

        train_loss = loss_train_sum / max(n_train_sum, 1)
        train_sc_epoch_report = reduce_reports(sc_report_all_train)
        train_sc_epoch_report.update({"split": "train", "epoch": int(epoch + 1), "n_samples": int(n_train_sum), "n_batches": int(len(sc_report_all_train))})
        sc_report_all_train.clear()

        train_gen_epoch_report = reduce_reports(gen_report_all_train)
        train_gen_epoch_report.update({"split": "train", "epoch": int(epoch + 1), "n_samples": int(n_train_sum), "n_batches": int(len(gen_report_all_train))})
        gen_report_all_train.clear()

        train_pw_epoch_report = reduce_reports(pw_report_all_train)
        train_pw_epoch_report.update({"split": "train", "epoch": int(epoch + 1), "n_samples": int(n_train_sum), "n_batches": int(len(pw_report_all_train))})
        pw_report_all_train.clear()

        train_disp_epoch_report = reduce_dispatch(disp_report_all_train)
        train_disp_epoch_report.update({"split": "train", "epoch": int(epoch + 1), "n_samples": int(n_train_sum), "n_batches": int(len(disp_report_all_train))})
        disp_report_all_train.clear()

        val_loss, val_sc_epoch_report, val_gen_epoch_report, val_pw_epoch_report, val_disp_epoch_report = eval_loss_on_scens(
            model, data_val, oats_dict, pg_lb, pg_ub, batch_size, device, dtype,
            PTDF_SC_full, G2B, W2B, D2B, Fmax_SC_full,
            split="val", epoch=int(epoch + 1), atol=atol, rtol=rtol, use_clip=use_clip)
        
        train_loss_hist.append(float(train_loss))
        val_loss_hist.append(float(val_loss))
        train_vio_hist.append(train_sc_epoch_report)
        val_vio_hist.append(val_sc_epoch_report)
        train_gen_hist.append(train_gen_epoch_report)
        val_gen_hist.append(val_gen_epoch_report)
        train_pw_hist.append(train_pw_epoch_report)
        val_pw_hist.append(val_pw_epoch_report)
        train_dispatch_hist.append(train_disp_epoch_report)
        val_dispatch_hist.append(val_disp_epoch_report)
        print(f"[Epoch {epoch+1}] train_loss={train_loss:.6e} val_loss={val_loss:.6e}")

    test_loss, test_vio, test_gen, test_pw, test_dispatch = eval_loss_on_scens(
        model, data_test, oats_dict, pg_lb, pg_ub, batch_size, device, dtype,
        PTDF_SC_full, G2B, W2B, D2B, Fmax_SC_full,
        split="test", epoch=int(epochs), atol=atol, rtol=rtol, use_clip=use_clip)
    test_loss_hist.append(float(test_loss))
    print(f"test_loss={test_loss:.6e}")

    db_path_obj = Path(db_path).resolve()
    base_dir = db_path_obj.parent
    out_dir = base_dir / db_path_obj.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "train_summary.xlsx"

    df_loss = pd.DataFrame({
        "epoch": np.arange(1, len(train_loss_hist) + 1),
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
    })

    df_train_vio = pd.DataFrame([dict(**r) for r in train_vio_hist])
    df_val_vio   = pd.DataFrame([dict(**r) for r in val_vio_hist])
    df_test_vio  = pd.DataFrame([dict(**test_vio)]) if isinstance(test_vio, dict) else pd.DataFrame([])
    df_train_gen = pd.DataFrame([dict(**r) for r in train_gen_hist])
    df_val_gen   = pd.DataFrame([dict(**r) for r in val_gen_hist])
    df_test_gen = pd.DataFrame([dict(**test_gen)]) if isinstance(test_gen, dict) else pd.DataFrame([])
    df_train_pw = pd.DataFrame([dict(**r) for r in train_pw_hist])
    df_val_pw   = pd.DataFrame([dict(**r) for r in val_pw_hist])
    df_test_pw  = pd.DataFrame([dict(**test_pw)]) if isinstance(test_pw, dict) else pd.DataFrame([])
    df_train_disp = pd.DataFrame(train_dispatch_hist)
    df_val_disp   = pd.DataFrame(val_dispatch_hist)
    df_test_disp  = pd.DataFrame([dict(**test_dispatch)]) if isinstance(test_dispatch, dict) else pd.DataFrame([])

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_loss.to_excel(writer, sheet_name="Loss", index=False)
        df_train_vio.to_excel(writer, sheet_name="Train_SC_Violation", index=False)
        df_val_vio.to_excel(writer, sheet_name="Val_SC_Violation", index=False)
        df_test_vio.to_excel(writer, sheet_name="Test_SC_Violation", index=False)
        df_train_gen.to_excel(writer, sheet_name="Train_GEN_Violation", index=False)
        df_val_gen.to_excel(writer, sheet_name="Val_GEN_Violation", index=False)
        df_test_gen.to_excel(writer, sheet_name="Test_GEN_Violation", index=False)
        df_train_pw.to_excel(writer, sheet_name="Train_PW_Violation", index=False)
        df_val_pw.to_excel(writer, sheet_name="Val_PW_Violation", index=False)
        df_test_pw.to_excel(writer, sheet_name="Test_PW_Violation", index=False)
        df_train_disp.to_excel(writer, sheet_name="Train_Dispatch", index=False)
        df_val_disp.to_excel(writer, sheet_name="Val_Dispatch", index=False)
        df_test_disp.to_excel(writer, sheet_name="Test_Dispatch", index=False)

    print("[Dim Check] Clean blocks consistent with testcase.")
    return model





if __name__ == "__main__":
    train_scopf_mlp()