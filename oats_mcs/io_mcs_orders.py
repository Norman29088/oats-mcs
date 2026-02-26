#==================================================================
# io_mcs_orders.py
# Storage I/O utilities for OATS-MCS dataset generation
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
# Last updated: 2026-02
#==================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal
import logging
import pandas as pd
import numpy as np
from pyomo.environ import value
from pyomo.core.base.var import VarData
io_logger = logging.getLogger(__name__)


IndexSet = Literal["G", "D", "L", "T", "W", "NONE"]


@dataclass(frozen=True)
class TableSpec:
    name: str
    order: List[str]

    src: str
    index_set: IndexSet

    scale_baseMVA: bool = True

    extras: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataOrder:
    model: str
    testcase: str
    sheets: Dict[str, str]

    # per-table specs: "PG" -> TableSpec(...)
    tables: Dict[str, TableSpec] = field(default_factory=dict)

    gen: List[str] = field(default_factory=list)
    dem: List[str] = field(default_factory=list)
    branch: List[str] = field(default_factory=list)
    transformer: List[str] = field(default_factory=list)
    wind: Optional[List[str]] = None


def sheet_exists(xl: pd.ExcelFile, sheet_name: str) -> bool:
    try:
        return sheet_name in xl.sheet_names
    except Exception:
        return False


def read_name_order(
    xl: pd.ExcelFile,
    sheet_name: str,
    name_col: str = "name",
    stat_col: str = "stat",
    active_only: bool = True,
    stat_active_value: int = 1,
) -> List[str]:
    if not sheet_exists(xl, sheet_name):
        return []

    df = xl.parse(sheet_name)
    if name_col not in df.columns:
        return []

    names = df[name_col].dropna().astype(str).map(str.strip)
    names = names[names != ""]

    if not active_only:
        return names.tolist()

    if stat_col not in df.columns:
        return names.tolist()

    stat = pd.to_numeric(df[stat_col], errors="coerce")

    active_mask = (stat == stat_active_value)
    active_mask = active_mask.reindex(names.index, fill_value=False)

    return names[active_mask].tolist()



def create_data_order(
    model: str,
    testcase: str,
    sheets: Optional[Dict[str, str]] = None,
    name_col: str = "name",
    wind_sheet: str = "wind",
    logger: Optional[logging.Logger] = None,
) -> DataOrder:
    lg = logger or io_logger

    default_sheets = {
        "generator": "generator",
        "demand": "demand",
        "branch": "branch",
        "transformer": "transformer",
        "wind": wind_sheet,
    }
    if sheets:
        default_sheets.update(sheets)
    sheets = default_sheets

    m = (model or "").strip()
    if not m:
        raise ValueError("model must be non-empty")

    with pd.ExcelFile(testcase) as xl:
        gen = read_name_order(xl, sheets["generator"], name_col=name_col, stat_col="stat", active_only=True)
        dem = read_name_order(xl, sheets["demand"],    name_col=name_col, stat_col="stat", active_only=True)
        branch = read_name_order(xl, sheets["branch"], name_col=name_col, stat_col="stat", active_only=True)
        transformer = read_name_order(xl, sheets["transformer"], name_col=name_col, stat_col="stat", active_only=True)
        if "wind" in xl.sheet_names: wind_list = read_name_order(xl, "wind", name_col=name_col, stat_col="stat", active_only=True) 
        elif "renewable" in xl.sheet_names: wind_list = read_name_order(xl, "renewable", name_col=name_col, stat_col="stat", active_only=True)
        else: wind_list = []
        wind = wind_list if len(wind_list) > 0 else None

    # build TableSpecs ("DataOrder.PG.order + mapping" idea)
    tables: Dict[str, TableSpec] = {}

    tables["PF_L"] = TableSpec(
        name="PF_L", order=list(branch), src="pL", index_set="L",
        scale_baseMVA=True, extras=["probC", "probOK"]
    )
    tables["PF_T"] = TableSpec(
        name="PF_T", order=list(transformer), src="pLT", index_set="T",
        scale_baseMVA=True
    )
   
    tables["PG"] = TableSpec(name="PG", order=gen, src="pG", index_set="G", scale_baseMVA=True)
    tables["PD"] = TableSpec(name="PD", order=dem, src="PD", index_set="D", scale_baseMVA=True)
    tables["PW"] = TableSpec(name="PW", order=wind, src="pW", index_set="W", scale_baseMVA=True)
    tables["PW_UB"] = TableSpec(name="PW_UB", order=wind, src="WGmax", index_set="W", scale_baseMVA=True)
    tables["PW_LB"] = TableSpec(name="PW_LB", order=wind, src="WGmin", index_set="W", scale_baseMVA=True)

    pfL = len(branch)
    pfT = len(transformer)

    lg.info(
        "[DataOrder] model=%s | gen=%d dem=%d branch=%d trafo=%d pfL=%d pfT=%d wind=%d | tables=%s",
        m, len(gen), len(dem), len(branch), len(transformer),
        pfL, pfT,
        len(wind) if wind else 0,
        list(tables.keys()),
    )

    return DataOrder(
        model=m,
        testcase=testcase,
        sheets=sheets,
        tables=tables,
        gen=gen,
        dem=dem,
        branch=branch,
        transformer=transformer,
        wind=wind,
    )


def merge_pf_LT(dfs: dict, keep_parts: bool = True) -> dict:
    """
    Merge PF_L + PF_T -> PF (columns: L then T).
    Extras (probC/probOK) are already on PF_L by design; they will remain in PF.
    """
    if "PF" in dfs:
        return dfs
    if ("PF_L" in dfs) and ("PF_T" in dfs):
        left  = dfs["PF_L"]
        right = dfs["PF_T"].drop(columns=["scenario_id"], errors="ignore")  # drop dup scenario;scenario->scenario_id
        dfs["PF"] = pd.concat([left, right], axis=1)
        if not keep_parts:
            dfs.pop("PF_L", None)
            dfs.pop("PF_T", None)
    return dfs


def read_and_validate_orders(testcase=None,
                            num_generators=None, num_demands=None, num_res=None):
    try:
        # --- generator / demand ---
        df_gen_order = pd.read_excel(testcase, sheet_name="generator")
        df_dem_order = pd.read_excel(testcase, sheet_name="demand")

        num_generators = len(df_gen_order)
        num_demands = len(df_dem_order)

        desired_res_order = None
        actual_num_res = 0
        try:
            try:
                df_res_order = pd.read_excel(testcase, sheet_name="wind")
            except Exception:
                df_res_order = pd.read_excel(testcase, sheet_name="renewable")

            desired_res_order = df_res_order["name"].astype(str).str.strip().tolist()
            actual_num_res = len(desired_res_order)

            if num_res is not None and num_res != actual_num_res:
                io_logger.info(f"Fix the number of RES from {num_res} to {actual_num_res}")
                num_res = actual_num_res
            elif num_res is None:
                num_res = actual_num_res

        except Exception:
            desired_res_order = None
            actual_num_res = 0
            num_res = 0

        return  num_generators, num_demands, num_res

    except Exception as e:
        io_logger.error(f"Error reading expected order from testcase: {e}", exc_info=True)
        raise



def prepare_scenario_inputs(
    scen,
    rand_list=None,
    dem_matrix=None,
    res_matrix=None,
):
    def _row(m):
        if m is None:
            return None
        try:
            if hasattr(m, "row"):          
                return list(m.row(scen))
            if hasattr(m, "iloc"):      
                return m.iloc[scen, :].to_list()
            return m[scen, :].tolist()   
        except Exception:
            return None

    dem_list  = _row(dem_matrix)
    res_list  = _row(res_matrix)
    return dem_list,  res_list

def align_round_num_and_truncate( 
    *,
    seed_num: int,
    round_num: int,
    model: str,
    dem_matrix=None,
    res_matrix=None,
    logger=None,
):
    """
    Align scenario count across all available matrices and truncate to the minimum row count.
    Returns: (round_num, dem_matrix, res_matrix, )
    """
    log = logger if logger is not None else logging

    mats = {
        "dem": dem_matrix,
        "res": res_matrix,
    }

    row_counts = []
    for k, m in mats.items():
        if m is None:
            continue
        try:
            row_counts.append(int(m.shape[0]))
        except Exception:
            row_counts.append(int(len(m)))

    if not row_counts:
        return round_num, dem_matrix, res_matrix

    orig_round = round_num
    min_round = min([orig_round, *row_counts])

    if min_round != orig_round or any(r != orig_round for r in row_counts):
        log.warning(
            f"In seed {seed_num}: before truncation - round_num={orig_round}, "
            f"rows={row_counts}. Truncating to {min_round}."
        )
        round_num = min_round

        def _cut(x):
            if x is None:
                return None
            if hasattr(x, "iloc"):
                return x.iloc[:min_round, :]
            return x[:min_round, :]

        dem_matrix = _cut(dem_matrix)
        res_matrix = _cut(res_matrix)


        log.info(f"In seed {seed_num}: after truncation - round_num={round_num}.")

    return round_num, dem_matrix, res_matrix
