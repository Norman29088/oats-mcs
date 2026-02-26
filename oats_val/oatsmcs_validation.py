#==================================================================
# oatsmcs_validation.py
# OATS-MCS Dataset Validation (Preventive SCOPF)
# Standalone execution script for dataset validation
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
# Last updated: 2026-02
#==================================================================

import numpy as np
import sqlite3
import pandas as pd
import os, torch
import scipy.sparse as sp
from tqdm import tqdm

OATSMCS_MAP = {
    "pG":    "Generator_Real",
    "PD":    "Demand_Real",
    "PF":    "PowerFlow",
    "PW":    "RES_Real",
    "PW_UB": "RES_UB",
    "PW_LB": "RES_LB",
}

def get_pg_bounds(oats_dict, device="cpu", dtype=torch.float64):
    gen = oats_dict["generator"]
    pg_lb = torch.as_tensor(gen["PGLB"].to_numpy(), dtype=dtype, device=device)
    pg_ub = torch.as_tensor(gen["PGUB"].to_numpy(), dtype=dtype, device=device)
    return pg_lb, pg_ub


def read_testcase_assets(xlsx_path):
    sheets = pd.read_excel(xlsx_path, sheet_name=["demand","generator","renewable","branch","transformer","bus"])
    line = pd.concat([sheets["branch"], sheets["transformer"]], ignore_index=True)
    return {
        "demand": sheets["demand"],
        "generator": sheets["generator"],
        "renewable": sheets["renewable"],
        "line": line,
        "bus": sheets["bus"]
    }

def read_slack_from_results(xlsx_path):
    bus = pd.read_excel(xlsx_path, sheet_name="bus")
    idx = bus["angle(degs)"].abs().idxmin()
    slack = int(bus.loc[idx, "name"]) - 1
    print(f"[Slack Bus] name={bus.loc[idx, 'name']}  zero_based={slack}")
    return slack

def read_dataset_sql(db_path: str, batch_size: int = None, batch_idx: int = 0) -> dict:
    data = {}
    with sqlite3.connect(db_path) as conn:
        for key, table_name in OATSMCS_MAP.items():
            query = f'SELECT * FROM "{table_name}" ORDER BY scenario_id'
            if batch_size is not None:
                offset = batch_idx * batch_size
                query += f' LIMIT {batch_size} OFFSET {offset}'
            data[key] = pd.read_sql_query(query, conn)
            print(f"Read {key}, shape <{data[key].shape}>")
    return data


def load_grid_params(grid_param_path, use_sc=False, device="cpu", dtype=torch.float64):
    pth = "grid_flow_params_sc.pth" if use_sc else "grid_flow_params_base.pth"
    obj = torch.load(os.path.join(grid_param_path, pth), map_location=device, weights_only=True)

    PTDF = obj["PTDF"].to(device=device, dtype=dtype); PTDF.requires_grad_(False)
    Fmax = obj["Fmax"].to(device=device, dtype=dtype); Fmax.requires_grad_(False)

    G2B_sp = sp.load_npz(os.path.join(grid_param_path, "G2B.npz"))
    D2B_sp = sp.load_npz(os.path.join(grid_param_path, "D2B.npz"))

    w2b_path = os.path.join(grid_param_path, "W2B.npz")
    W2B_sp = sp.load_npz(w2b_path) if os.path.exists(w2b_path) else None

    G2B = torch.tensor(G2B_sp.toarray(), dtype=dtype, device=device); G2B.requires_grad_(False)
    D2B = torch.tensor(D2B_sp.toarray(), dtype=dtype, device=device); D2B.requires_grad_(False)
    W2B = torch.tensor(W2B_sp.toarray(), dtype=dtype, device=device) if W2B_sp is not None else None
    if W2B is not None: W2B.requires_grad_(False)

    wshape = tuple(W2B.shape) if W2B is not None else None
    print(f"\n[Grid Params] PTDF{tuple(PTDF.shape)} Fmax{tuple(Fmax.shape)} G2B{tuple(G2B.shape)} W2B{wshape} D2B{tuple(D2B.shape)}")

    return PTDF, Fmax, G2B, W2B, D2B



def validation_check(data):
    # Step 1 — sovler status check
    valid_mask = (data["pG"]["solver_status"] == "ok") &(data["pG"]["term_cond"] == "optimal")
    n_total = len(data["pG"])
    n_valid = valid_mask.sum()
    print(f"\n[Solver Status Check] total samples = {n_total}, valid samples = {n_valid}")

    if n_total != n_valid:
        raise ValueError(f"[Solver Status Check] Invalid scenarios detected: {n_total - n_valid}")
    print("[Solver Status Check] All scenarios are OK & optimal.")

    # Step 2 — scenario_id match validation
    id_ref = set(data["PD"]["scenario_id"])

    for k in data:
        if set(data[k]["scenario_id"]) != id_ref:
            raise ValueError(f"[ID Check] {k} scenario_id mismatch.")

    print("[ID Check] All dataset aligned.")



def clean_and_dim_check(data, assets):
    data["pG"] = data["pG"].drop(columns=["scenario_id","scen","solver_status","term_cond","PG_Total","PR_Total","PW_Total","PD_Total","P_Diff"], errors="ignore")
    data["PD"] = data["PD"].drop(columns=["scenario_id","scen","PD_Total"], errors="ignore")  
    data["PF"] = data["PF"].drop(columns=["scenario_id","scen","probC","probOK"], errors="ignore")
    data["PW"] = data["PW"].drop(columns=["scenario_id","scen","PW_Total","PR_Total"], errors="ignore")
    data["PW_UB"] = data["PW_UB"].drop(columns=["scenario_id","scen"], errors="ignore")
    data["PW_LB"] = data["PW_LB"].drop(columns=["scenario_id","scen"], errors="ignore")

    if data["pG"].shape[1] != len(assets["generator"]): raise ValueError("Generator dimension mismatch.")
    if data["PD"].shape[1] != len(assets["demand"]): raise ValueError("Demand dimension mismatch.")
    if data["PW"].shape[1] != len(assets["renewable"]): raise ValueError("Renewable dimension mismatch.")
    if data["PW_UB"].shape[1] != len(assets["renewable"]): raise ValueError("Renewable UB dimension mismatch.")
    if data["PW_LB"].shape[1] != len(assets["renewable"]): raise ValueError("Renewable LB dimension mismatch.")
    if data["PF"].shape[1] != len(assets["line"]): raise ValueError("Line flow dimension mismatch.")

    print("\n[Dim Check] Clean blocks consistent with testcase.")
    return data


# def to_torch_blocks(data, keys, device="cpu", dtype=torch.float64):
#     for k in keys:
#         if isinstance(data[k], pd.DataFrame):
#             data[k] = torch.tensor(data[k].to_numpy(dtype=np.float64), device=device, dtype=dtype)
#     return datafv

def to_torch_blocks(data, keys, device="cpu", dtype=torch.float64):
    for key in keys:
        dataframe = data[key]
        # numpy_array = dataframe.to_numpy(copy=False)
        numpy_array = dataframe.to_numpy(dtype=np.float64, copy=False)
        tensor = torch.as_tensor(numpy_array, dtype=dtype)
        if str(device) != "cpu": tensor = tensor.to(device, non_blocking=True)
        data[key] = tensor
    return data

def power_balance_check(data, atol=1e-6):
    balance = data["pG"].sum(dim=1) + data["PW"].sum(dim=1) - data["PD"].sum(dim=1)
    if not torch.allclose(balance, torch.zeros_like(balance), atol=atol, rtol=0.0):
        max_err = float(balance.abs().max().item())
        raise ValueError(f"[Power Balance Check] Power mismatch. Max error = {max_err}")
    print("\n[Power Balance Check] DC power balance satisfied.")


def power_flow_dc(data, PTDF, G2B, W2B, D2B):
    Pinj_bus = (data["pG"] @ G2B.T) + (data["PW"] @ W2B.T) - (data["PD"] @ D2B.T)
    return Pinj_bus @ PTDF.T        # (N, n_line)

def power_flow_sc(data, PTDF_sc, G2B, W2B, D2B):
    Pinj_bus = (data["pG"] @ G2B.T) + (data["PW"] @ W2B.T) - (data["PD"] @ D2B.T)   # (N,n_bus)
    # PTDF_sc: (C,line,n_bus)  -> output (N,C,line)
    return torch.einsum("nb,clb->ncl", Pinj_bus, PTDF_sc)

def fmax_violation(PF, Fmax, atol=1e-6, rtol=0.0):
    PF = torch.as_tensor(PF)
    Fmax = torch.as_tensor(Fmax, dtype=PF.dtype, device=PF.device)

    if PF.ndim == 2:
        # PF: (N,L), Fmax: (L,)
        limit = Fmax * (1.0 + rtol) + atol
        over = torch.abs(PF) - limit.view(1, -1)
        return torch.clamp(over, min=0.0)

    if PF.ndim == 3:
        # PF: (N,C,L)
        if Fmax.ndim == 1:
            limit = (Fmax * (1.0 + rtol) + atol).view(1, 1, -1)      # (1,1,L)
        elif Fmax.ndim == 2:
            limit = (Fmax * (1.0 + rtol) + atol).view(1, Fmax.shape[0], Fmax.shape[1])  # (1,C,L)
        else:
            raise ValueError("Fmax must be (L,) or (C,L) for PF.ndim==3")

        over = torch.abs(PF) - limit
        return torch.clamp(over, min=0.0)

    raise ValueError("PF must be (N,L) or (N,C,L)")

def print_violation_stats(vio, name="vio"):
    # vio: (N,L) or (N,C,L)
    vmax = float(vio.max().item())
    vsum = float(vio.sum().item())
    print(f"[{name}] Flow Violations: max={vmax:.6e} sum={vsum:.6e} shape={tuple(vio.shape)}")
    return vmax,vsum


def build_line_incidence_A_reduced(n_bus, from_bus, to_bus, slack, device="cpu", dtype=torch.float64):
    fb = torch.as_tensor(from_bus, dtype=torch.long, device=device)
    tb = torch.as_tensor(to_bus,   dtype=torch.long, device=device)
    n_line = int(fb.numel())

    A_full = torch.zeros((n_bus, n_line), dtype=dtype, device=device)
    cols = torch.arange(n_line, device=device)
    A_full[fb, cols] =  1.0
    A_full[tb, cols] = -1.0

    mask = torch.ones(n_bus, dtype=torch.bool, device=device)
    mask[slack] = False
    return A_full[mask, :]


def build_lodf_from_ptdf_reduced(PTDF_red, from_bus, to_bus, slack, n_bus,out_k_all=None, device="cpu", eps=1e-12):
    """
    PTDF_red: (n_line, n_bus-1) (slack column removed)
    out_k_all: list of contingency line indices (0-based).
    """
    n_line, n_bus_1 = PTDF_red.shape
    assert n_bus_1 == n_bus - 1

    A_red = build_line_incidence_A_reduced(n_bus, from_bus, to_bus, slack, device=device, dtype=torch.float64)
    H = PTDF_red @ A_red  # (n_line, n_line)

    den = 1.0 - torch.diag(H) # (n_line,)
    k_check = torch.as_tensor(out_k_all, device=PTDF_red.device, dtype=torch.long)
    if k_check.ndim != 1:
        raise ValueError("out_k_all must be a 1D list/array of line indices")
    
    mask = den[k_check].abs() < eps
    bad = k_check[mask]
    if len(bad) > 0:
       raise ValueError(f"[LODF] near-singular denom at lines={bad[:20].tolist()} (1-Hkk ~ 0), eps={eps:.1e}")

    LODF = H / den[None, :]
    LODF.fill_diagonal_(-1.0)
    return LODF

def build_pf_sc_from_lodf(PF_dc, LODF, out_k_all):
    N, n_line = PF_dc.shape
    C = len(out_k_all)
    PF_sc_lodf = torch.empty((N,C,n_line), device=PF_dc.device, dtype=PF_dc.dtype)
    for c_idx, k in enumerate(out_k_all):
        PF_sc_lodf[:, c_idx, :] = PF_dc + PF_dc[:, [k]] @ LODF[:, [k]].T
    return PF_sc_lodf

def torch_p95_abs(x_abs: torch.Tensor) -> float:
    if x_abs.numel() == 0:
        return 0.0
    k = max(1, int((95 * x_abs.numel() + 99) // 100))
    return float(x_abs.flatten().kthvalue(k)[0])


def bound_violation_analysis(
    pred: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    *,
    split: str,
    tag: str,
    verbose: bool = False,
):
    if tag == "gen":
        lb = lb.reshape(1, -1) 
        ub = ub.reshape(1, -1)

    vio_lb = torch.clamp(lb - pred, min=0.0)
    vio_ub = torch.clamp(pred - ub, min=0.0)
    vio = vio_lb + vio_ub
    vio = torch.where(vio.abs() < 1e-12, torch.zeros_like(vio), vio)

    return check_consistency_residual_analysis(
        vio, torch.zeros_like(vio), atol=0.0, rtol=0.0,
        name=f"{split}_{tag}_violation", verbose=verbose
    )

def gen_bound_violation_analysis(pred_pg, pg_lb, pg_ub, *, split: str, verbose: bool = False):
    return bound_violation_analysis(pred_pg, pg_lb, pg_ub, split=split, tag="gen", verbose=verbose)

def pw_bound_violation_analysis(pred_pw, pw_lb, pw_ub, *, split: str, verbose: bool = False):
    return bound_violation_analysis(pred_pw, pw_lb, pw_ub, split=split, tag="pw", verbose=verbose)

def check_consistency_residual_analysis(
    pred: torch.Tensor,
    ref: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    name: str = "consistency",
    verbose: bool = True
) -> dict:

    diff = pred - ref
    absd = diff.abs()
    thr = atol + rtol * ref.abs()
    bad = absd > thr

    bad_cnt = int(bad.sum().item())
    max_abs = float(absd.max().item()) if absd.numel() else 0.0
    mean_abs = float(absd.mean().item()) if absd.numel() else 0.0
    p95_abs = torch_p95_abs(absd)

    if verbose:
        print(
            f"\n[Residual Analysis: {name}]\n"
            f"  Criterion            : |pred - ref| <= atol + rtol * |ref|\n"
            f"  Absolute Tolerance   : {atol:.2e}\n"
            f"  Relative Tolerance   : {rtol:.2e}\n"
            f"  Total Elements       : {absd.numel()}\n"
            f"  Tensor Shape         : {tuple(pred.shape)}\n"
            f"  Violations Count     : {bad_cnt}\n"
            f"  Max Absolute Diff    : {max_abs:.6e}\n"
            f"  Mean Absolute Diff   : {mean_abs:.6e}\n"
            f"  95th Percentile Diff : {p95_abs:.6e}\n"
        )

    if verbose and bad_cnt > 0:
        idx = torch.nonzero(bad, as_tuple=False)[0].tolist()
        loc = ",".join([str(v) for v in idx])
        t = tuple(idx)
        print(
            f"  First Violation Index: ({loc})\n"
            f"  Predicted Value      : {float(pred[t].item()):.6e}\n"
            f"  Reference Value      : {float(ref[t].item()):.6e}\n"
            f"  Absolute Difference  : {float(diff[t].item()):.6e}\n"
            f"  Threshold            : {float(thr[t].item()):.6e}\n"
        )

    return {
        "analysis_name": name,
        "violation_count": bad_cnt,
        "maximum_absolute_difference": max_abs,
        "mean_absolute_difference": mean_abs,
        "percentile95_absolute_difference": p95_abs,
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "total_number_of_elements": int(absd.numel()),
        "tensor_shape": tuple(pred.shape),
    }




def load_and_validate_dataset(
    testcase="IEEE_DataPort/case24_ieee_rts_dtu_dataport.xlsx",
    batch_size=None,
    resultcase="IEEE_DataPort/results.xlsx",
    db_path="IEEE_DataPort/merged_scene_21.db",
    grid_param_path="IEEE_DataPort/N-1_contingency_test",
    atol=1e-5,
    rtol=1e-5,
    device="cpu",
    dtype=torch.float64
):
    # Step3 read testcase + dim check (fixed resources loaded once)
    oats_dict = read_testcase_assets(testcase)

    # Step5 power flow check (fixed grid params loaded once)
    PTDF, Fmax, G2B, W2B, D2B = load_grid_params(grid_param_path, use_sc=False, device=device, dtype=dtype)
    PTDF_SC_full, Fmax_SC_full, _, _, _ = load_grid_params(grid_param_path, use_sc=True, device=device, dtype=dtype)
    PTDF_SC = PTDF_SC_full[1:, :]
    Fmax_SC = Fmax_SC_full[1:, :]
    pg_lb, pg_ub = get_pg_bounds(oats_dict, device=device, dtype=dtype)  # (nG,), (nG,)


    cflag = oats_dict["line"]["contingency"].to_numpy().astype(int)
    out_k_all = np.where(cflag != 0)[0].astype(int).tolist()
    print(f"[N-1] out_k_all count={len(out_k_all)} head={out_k_all[:50]}")

    n_bus = len(oats_dict["bus"]["name"])
    from_bus = oats_dict["line"]["from_busname"].to_numpy().astype(int) - 1
    to_bus = oats_dict["line"]["to_busname"].to_numpy().astype(int) - 1
    slack = read_slack_from_results(resultcase)
    LODF = build_lodf_from_ptdf_reduced(PTDF, from_bus, to_bus, slack, n_bus,out_k_all, device=device, eps=1e-12)

    batch_idx = 0
    dc_report_all = []
    sc_report_all = []
    dc_vio_report_all,sc_vio_report_all = [] ,[]
    gen_vio_report_all,pw_vio_report_all = [], []

    if batch_size is not None:
        with sqlite3.connect(db_path) as conn:
            total = conn.execute('SELECT COUNT(*) FROM "Generator_Real"').fetchone()[0]

        n_batch = (total + batch_size - 1) // batch_size
        pbar = tqdm(total=n_batch, desc="Validating batches", position=0, leave=True,dynamic_ncols=True)
    else:
        pbar = None

    while True:
        # Step 0 READ OATSMCS dataset.
        data = read_dataset_sql(db_path, batch_size=batch_size, batch_idx=batch_idx)

        if batch_size is not None and len(data["pG"]) == 0:
            break

        # Step 1 — solver status check & Step 2 — scenario_id match validation
        validation_check(data)

        # Step3 read testcase + dim check -> torch format
        data = clean_and_dim_check(data, oats_dict)
        data = to_torch_blocks(data, keys=["pG","PD","PW","PF","PW_UB","PW_LB"], device=device, dtype=dtype)


        # Step4 power balance check
        power_balance_check(data, atol=1e-6)

        # Step5 power flow check
        # DC: PTDF (PF_dc) vs Pyomo base PF (data["PF"]:reference)
        # Base PF
        PF_dc = power_flow_dc(data, PTDF, G2B, W2B, D2B)
        vio_dc = fmax_violation(PF_dc, Fmax, atol=atol)
        print_violation_stats(vio_dc, "violation_dc")

        zero_dc = torch.zeros_like(vio_dc)
        dc_vio_report = check_consistency_residual_analysis(vio_dc, zero_dc,atol=0, rtol=0,name="dc_flow_violation", verbose=False)
        dc_vio_report_all.append(dc_vio_report)

        # DC consistency
        dc_report = check_consistency_residual_analysis(PF_dc, data["PF"],atol=atol, rtol=rtol,name="dc_ptdf_vs_pyomo", verbose=False)
        dc_report_all.append(dc_report)


        # SC-PTDF PF
        PF_sc_ptdf = power_flow_sc(data, PTDF_SC, G2B, W2B, D2B)
        vio_sc = fmax_violation(PF_sc_ptdf, Fmax_SC, atol=atol)
        print_violation_stats(vio_sc, "violation_sc")

        zero_sc = torch.zeros_like(vio_sc)
        sc_vio_report = check_consistency_residual_analysis(vio_sc, zero_sc,atol=0, rtol=0,name="sc_flow_violation", verbose=False)
        sc_vio_report_all.append(sc_vio_report)

        # SC consistency
        PF_sc_lodf = build_pf_sc_from_lodf(PF_dc, LODF, out_k_all)
        sc_report = check_consistency_residual_analysis(PF_sc_ptdf, PF_sc_lodf,atol=atol, rtol=rtol,name="scptdf_vs_lodf", verbose=False)
        sc_report_all.append(sc_report)

        # Generator bound violation (pG vs [pg_lb, pg_ub]) 
        gen_vio_report = gen_bound_violation_analysis(data["pG"], pg_lb, pg_ub, split="dataset", verbose=False)
        gen_vio_report_all.append(gen_vio_report)

        pw_vio_report = pw_bound_violation_analysis(data["PW"], data["PW_LB"], data["PW_UB"], split="dataset", verbose=False)
        pw_vio_report_all.append(pw_vio_report)

        del PF_sc_ptdf, PF_sc_lodf, PF_dc, data
        batch_idx += 1
        if batch_size is None:
            break
    pbar.close()

    total_viol_dc = sum(r["violation_count"] for r in dc_report_all)
    max_diff_dc = max(r["maximum_absolute_difference"] for r in dc_report_all)
    mean_diff_dc = sum(r["mean_absolute_difference"] for r in dc_report_all) / max(len(dc_report_all),1)

    print("\n[DC Summary]")
    print("Total Violations:", total_viol_dc)
    print("Global Max Abs Diff:", max_diff_dc)
    print("Mean Abs Diff (batch avg):", mean_diff_dc)

    total_viol_sc = sum(r["violation_count"] for r in sc_report_all)
    max_diff_sc = max(r["maximum_absolute_difference"] for r in sc_report_all)
    mean_diff_sc = sum(r["mean_absolute_difference"] for r in sc_report_all) / max(len(sc_report_all),1)

    print("\n[SC Summary]")
    print("Total Violations:", total_viol_sc)
    print("Global Max Abs Diff:", max_diff_sc)
    print("Mean Abs Diff (batch avg):", mean_diff_sc)

    total_viol_g = sum(r["violation_count"] for r in gen_vio_report_all)
    max_diff_g = max(r["maximum_absolute_difference"] for r in gen_vio_report_all) if gen_vio_report_all else 0.0
    mean_diff_g = sum(r["mean_absolute_difference"] for r in gen_vio_report_all) / max(len(gen_vio_report_all), 1)

    print("\n[GEN Summary]")
    print("Total Violations:", total_viol_g)
    print("Global Max Abs Diff:", max_diff_g)
    print("Mean Abs Diff (batch avg):", mean_diff_g)

    total_viol_pw = sum(r["violation_count"] for r in pw_vio_report_all)
    max_diff_pw = max(r["maximum_absolute_difference"] for r in pw_vio_report_all) if pw_vio_report_all else 0.0

    print("\n[PW Summary]")
    print("Total Violations:", total_viol_pw)
    print("Global Max Abs Diff:", max_diff_pw)

    return PTDF_SC_full, dc_report_all, sc_report_all, dc_vio_report_all, sc_vio_report_all, oats_dict, gen_vio_report_all,pw_vio_report_all


if __name__ == "__main__":

    load_and_validate_dataset()
