#==================================================================
# data_generator.py
# Dataset generation module for OATS-MCS
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
#==================================================================

import numpy as np
import pandas as pd
import polars as pl
import sqlite3
import os
from scipy.stats import qmc  
from oats_mcs.io_mcs_storage import HDF5_KEY_MAP
from numpy.random import SeedSequence

def read_data_from_file(file_path, sheet_name, dtype=np.float64, index_col=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=index_col)
    df = df.astype(dtype)
    return df, df.shape[1]

def read_data_from_csv(file_path, dtype=np.float64, index_col=0):
    df = pd.read_csv(file_path, index_col=index_col)
    df = df.astype(dtype)
    return df, df.shape[1]

def read_data_from_sql(db_path, table_name, dtype=np.float64, index_col=0):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, index_col=index_col)
    df = df.astype(dtype)
    return df, df.shape[1]

def write_data_to_file(data, out_path, sheet_name, var_prefix, num_variable, num_scen, dtype=np.float64):
    df = pd.DataFrame(
        data,
        columns=[f'{var_prefix}_{i}' for i in range(num_variable)],
        index=[f'Scen_{i}' for i in range(num_scen)]
    )
    csv_path = out_path.replace(".xlsx", ".csv")
    df.to_csv(csv_path)
    print("Data saved to CSV at", csv_path)


def _sobol_unit_block(num_scen: int, num_cols: int, sobol_seed: int, start: int, dtype=np.float64) -> np.ndarray:
    sampler = qmc.Sobol(d=num_cols, scramble=True, seed=int(sobol_seed))
    if start > 0:
        sampler.random(n=int(start))  # discard to reach the desired offset
    return sampler.random(n=int(num_scen)).astype(dtype)


def _split_seed_block(rand_seed: int) -> tuple[int, int]:
    """
    Split rand_seed into (seed_main, block_id) using the last 3 digits as block id.

    Recommended convention (e.g., 6 digits):
    - seed_main = rand_seed // 1000   (e.g., 123456 -> 123)
    - block_id  = rand_seed % 1000    (e.g., 123456 -> 456)

    Notes:
    - rand_seed has no digit limit; only the last 3 digits are reserved for block_id.
    - If rand_seed < 1000, treat as a single block: (seed_main=rand_seed, block_id=0).
    """
    rs = int(rand_seed)
    if rs < 1000:
        return rs, 0
    return rs // 1000, rs % 1000

def _rng_block(rand_seed: int):
    """
    Deterministic per-block RNG, compatible with parallel block generation.
    """
    seed_main, block_id = _split_seed_block(rand_seed)
    ss = SeedSequence([int(seed_main), int(block_id)])
    return np.random.default_rng(ss)


def generate_random_matrix(num_scen, num_cols, expect_range, distribution='uniform',
                           rand_seed=1234, dtype=np.float64):
    # np.random.seed(rand_seed)
    rng = np.random.default_rng(rand_seed) # Using local RNG
    low, high = expect_range

    if distribution == 'multi_lhs':
        sampler = qmc.LatinHypercube(d=num_cols, seed=rng) 
        unit_samples = sampler.random(n=num_scen).astype(dtype)   # shape (N, d)

        #Linear mapping to [low, high]
        matrix = low + (high - low) * unit_samples

    elif distribution.startswith("lhs_"):
        # e.g., distribution="lhs_100000"
        try:
            lhs_total = int(distribution.split("_", 1)[1])
        except Exception as e:
            raise ValueError(f"Invalid LHS total size in distribution='{distribution}'. Use 'lhs_<N>'") from e

        seed_main, block_id = _split_seed_block(rand_seed)

        sampler = qmc.LatinHypercube(d=num_cols, seed=np.random.default_rng(int(seed_main)))
        unit_all = sampler.random(n=int(lhs_total)).astype(dtype)

        start = int(block_id) * int(num_scen)
        end = start + int(num_scen)
        if end > int(lhs_total):
            raise ValueError(
                f"LHS slice out of range: block_id={block_id}, num_scen={num_scen}, "
                f"lhs_total={lhs_total} -> slice [{start}:{end}) exceeds total."
            )

        unit_samples = unit_all[start:end, :]
        del unit_all

        matrix = low + (high - low) * unit_samples

    elif distribution == 'multi_uniform': 
        matrix = rng.uniform(low, high, size=(num_scen, num_cols)).astype(dtype) 

    elif distribution == 'uniform':
        seed_main, block_id = _split_seed_block(rand_seed)
        rng_main = np.random.default_rng(int(seed_main))
        offset = int(block_id) * int(num_scen) * int(num_cols)
        if offset > 0:
            rng_main.random(offset)  

        unit = rng_main.random(int(num_scen) * int(num_cols)).reshape(int(num_scen), int(num_cols)).astype(dtype)
        matrix = low + (high - low) * unit

    elif distribution == 'sobol':
        seed_main, block_id = _split_seed_block(rand_seed)
        sobol_seed = int(seed_main)                 # scramble seed version
        start = int(block_id) * int(num_scen)       # block offset
        unit_samples = _sobol_unit_block(num_scen, num_cols, sobol_seed=sobol_seed, start=start, dtype=dtype)
        matrix = low + (high - low) * unit_samples

    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    return matrix


#-----------------------------#------------------#-------------------------------


def generate_or_read_data(
    num_scen,
    num_variable,
    expect_range=(0.95, 1.05),
    dtype=np.float64,
    var_prefix="Variable",
    distribution='uniform',
    write_to_sql=False,
    save_to_excel=False,
    out_sql_table=None,
    out_excel_path=None,
    out_sql_path=None,
    rand_seed=1234,
    baseline_path=None,     
    baseline_sheet=None,    
    baseline_col=None       
):
    data = None
    rand_seed_save =int(rand_seed)
    vp = var_prefix.lower()
    if vp in ("demand", "pd"):
        HDF5_KEY = "PD"
        rand_seed=rand_seed
    elif vp in ("res", "wind","renewable"):
        HDF5_KEY = "PW_UB"
        rand_seed=rand_seed+15000 #Disturbance on random seed for RES

    matrix = generate_random_matrix(num_scen, num_variable, expect_range, distribution, rand_seed, dtype)
    data = pl.DataFrame(matrix)
    if baseline_path is not None:
        if vp in ("demand", "pd"):
            baseline_sheet = "demand"
        elif vp in ("res", "wind","renewable"):
            sheet_names = pd.ExcelFile(baseline_path).sheet_names # Read sheet names:
            if "wind" in sheet_names:baseline_sheet = "wind"
            elif "renewable" in sheet_names:baseline_sheet = "renewable"
            else:raise ValueError(f"No wind/renewable sheet found. Available sheets: {sheet_names}")
        else:
            raise ValueError(f"Cannot infer baseline_sheet from var_prefix='{var_prefix}'.")
        if baseline_col is None:
            if vp in ("demand","pd"):
                baseline_col = "real"
            elif vp in ("res","wind","pw","renewable"):
                baseline_col = "PGUB"
            else:
                raise ValueError(f"Cannot infer baseline_col from var_prefix='{var_prefix}'.")
                
        base_df = pd.read_excel(baseline_path, sheet_name=baseline_sheet)
        base_df_name = base_df["name"].astype(str).tolist()
        if baseline_col not in base_df.columns:
            raise ValueError(f"Column '{baseline_col}' not found in sheet '{baseline_sheet}'.")
        baseline = base_df[baseline_col].astype(dtype).values
        if baseline.shape[0] != num_variable:
            raise ValueError(f"Baseline length({baseline.shape[0]}) != num_variable({num_variable}).")
        
        baseline = [(pl.col(col) * float(baseline[i])).alias(col)for i, col in enumerate(data.columns)]
        data = data.with_columns(baseline)
        print("Baseline applied: random coefficients * baseline values.")
    else:
        base_df_name = [f"x{i}" for i in range(int(num_variable))]
    data = data.to_numpy() if isinstance(data, pl.DataFrame) else data
    df_data = None
    if data.shape[1] > 2000:  write_to_sql = False

    if write_to_sql or save_to_excel:
        scenario_id = [f"s{int(rand_seed_save):06d}_{int(s):06d}" for s in range(0, int(num_scen))]
        df_data = pd.DataFrame(data, columns=base_df_name)
        df_data.insert(0, "scenario_id", scenario_id)

    if write_to_sql:
        TABLE_MAP = {**HDF5_KEY_MAP, "PG_UB": "Generator_UB"}
        table_name = TABLE_MAP[HDF5_KEY]
        base_dir = os.path.join("OATS_MCS_result", f"Data_sampling")
        os.makedirs(base_dir, exist_ok=True)
        db_path = os.path.join(base_dir, f"oats_scens_n{int(num_scen)}_seed{int(rand_seed_save)}.db")

        with sqlite3.connect(db_path) as conn:
            df_data.to_sql(table_name, conn, if_exists="append", index=True, index_label="scen")
        print(f"SAVE SQL {table_name} rows={df_data.shape[0]} cols={df_data.shape[1]} -> {db_path}")

    if save_to_excel:
        csv_path = out_excel_path if out_excel_path else f"generated_{var_prefix}.csv"
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df_data.to_csv(csv_path, index=False)
        print(f"SAVE CSV rows={df_data.shape[0]} cols={df_data.shape[1]} -> {csv_path}")

    if df_data is not None:
        del df_data

    return data


def get_dataset(dataset_type, num_scen, num_variable, **kwargs):
    return generate_or_read_data(num_scen, num_variable, var_prefix=dataset_type, **kwargs)

