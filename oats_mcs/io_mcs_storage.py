#==================================================================
# io_mcs_storage.py
# Storage I/O utilities for OATS-MCS dataset generation
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
#==================================================================

import os
import logging,sqlite3
import pandas as pd
import hashlib


io_logger = logging.getLogger(__name__)
DEFAULT_TABLE_ORDER = ["PG","PD","PF", "PW","PW_UB","PW_LB",]

HDF5_KEY_MAP = {
    "PG": "Generator_Real",
    "PD": "Demand_Real",
    "PF": "PowerFlow",

    "PW": "RES_Real",
    "PW_UB": "RES_UB",
    "PW_LB": "RES_LB",
}
#------------------------------------------------------
def calculate_hashes(file_path):
    """Calculate the MD5 and SHA256 values ​​of the specified file."""
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # 4096 bytes = 4KB, reading 4KB at a time is suitable for most files, even 10MB only requires 2560 iterations.
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    
    return md5_hash.hexdigest(), sha256_hash.hexdigest()
#------------------------------------------------------

def assert_scen_aligned(*dfs):
    base = None
    for df in dfs:
        if df is None:
            continue

        assert df.index.is_unique, "Scenario index is not unique"

        if base is None:
            base = df.index
        else:
            assert base.equals(df.index), "Scenario index mismatch between tables"
#------------------------------------------------------


def save_to_db(
    *,
    dfs: dict | None = None,
    db_path="oats_scenarios.db",
    oats_order=False, resultcase=None,
    generator_sheet="generator", demand_sheet="demand", branch_sheet="branch",
    max_total_cols=2000,
    if_exists="replace",
    return_hash: bool = False,
):
    # unpack dfs
    pg_df = pd_df = pf_df = pw_df = None
    pw_ub_df = pw_lb_df = None
    if dfs is not None:
        pg_df    = dfs.get("PG", pg_df)
        pd_df    = dfs.get("PD", pd_df)
        pf_df    = dfs.get("PF", pf_df)
        pw_df    = dfs.get("PW", pw_df)
        pw_ub_df = dfs.get("PW_UB", None)
        pw_lb_df = dfs.get("PW_LB", None)

    if pg_df is None or pd_df is None or pf_df is None:
        raise ValueError("save_to_db requires PG/PD/PF (missing one of them).")

    if oats_order:
        if resultcase is None:
            raise ValueError("When oats_order is True, resultcase must be provided.")
        try:
            gen_order = pd.read_excel(resultcase, sheet_name=generator_sheet)["name"].tolist()
            pg_df = pg_df[gen_order]

            dem_order = pd.read_excel(resultcase, sheet_name=demand_sheet)["name"].tolist()
            pd_df = pd_df[dem_order]

            try:
                pf_order = pd.read_excel(resultcase, sheet_name=branch_sheet)["name"].tolist()
                extra_cols = [col for col in pf_df.columns if col not in pf_order]
                pf_df = pf_df[pf_order + extra_cols]
            except Exception as e:
                io_logger.error(f"Error in reordering PF columns: {e}", exc_info=True)

            io_logger.info("Reordered PG/PD/PF based on resultcase order.")
        except Exception as e:
            io_logger.error(f"Error in reordering columns: {e}", exc_info=True)

    assert_scen_aligned(pg_df, pd_df, pf_df, pw_df, pw_lb_df,pw_ub_df)

    try:
        with sqlite3.connect(db_path) as conn:
            pg_df.to_sql("Generator_Real", conn, if_exists=if_exists, index=True, index_label="scen")
            pd_df.to_sql("Demand_Real",    conn, if_exists=if_exists, index=True, index_label="scen")
            pf_df.to_sql("PowerFlow",      conn, if_exists=if_exists, index=True, index_label="scen")
            if pw_df is not None:
                pw_df.to_sql("RES_Real", conn, if_exists=if_exists, index=True, index_label="scen")
            if pw_ub_df is not None:
                pw_ub_df.to_sql("RES_UB", conn, if_exists=if_exists, index=True, index_label="scen")
            if pw_lb_df is not None:
                pw_lb_df.to_sql("RES_LB", conn, if_exists=if_exists, index=True, index_label="scen")

        io_logger.info(f"SQLite write OK: {db_path}")
        print(f"SQLite write OK: {db_path}")

        if return_hash:
            try:
                md5, sha256 = calculate_hashes(db_path)
                io_logger.info(f"SQLite hash OK: md5={md5}, sha256={sha256}")
                return md5, sha256
            except Exception as e:
                io_logger.error(f"SQLite hash failed: {e}", exc_info=True)
                return "Error", "Error"

        return None

    except Exception as e:
        io_logger.error(f"Error saving to DB: {e}", exc_info=True)
        if return_hash:
            return "Error", "Error"
        raise
#----------------------------------------------------------------------------------------------------------------------
def save_to_parquet(df, filepath):
    try:
        df.to_parquet(filepath)
        print(f"Saved DataFrame to parquet: {filepath}")
        io_logger.info(f"Saved DataFrame to parquet: {filepath}")
    except Exception as e:
        print(f"Error saving to parquet: {e}")
        io_logger.error(f"Error saving to parquet: {e}")
#------------------------------------------------------

def save_to_hdf5(df, filepath, key='data',mode='w'):
    try:
        df.to_hdf(filepath, key=key, mode=mode)
        print(f"Saved DataFrame to hdf5: {filepath}")
        io_logger.info(f"Saved DataFrame to hdf5: {filepath}")
    except Exception as e:
        print(f"Error saving to hdf5: {e}")
        io_logger.error(f"Error saving to hdf5: {e}")
#------------------------------------------------------

def save_to_csv(df, filepath):
    try:
        df.to_csv(filepath, index=False)
        print(f"Saved DataFrame to csv: {filepath}")
        io_logger.info(f"Saved DataFrame to csv: {filepath}")
    except Exception as e:
        print(f"Error saving to csv: {e}")
        io_logger.error(f"Error saving to csv: {e}")
#------------------------------------------------------

def save_to_hdf5_bundle(
    dfs: dict,
    hdf5_filename: str,
    *,
    return_hash: bool = False,
    include: list | None = None,
):
    keys = DEFAULT_TABLE_ORDER if include is None else list(include)

    first = True
    try:
        for k in keys:
            df = dfs.get(k)
            if df is None:
                continue
            h5_key = HDF5_KEY_MAP[k]
            save_to_hdf5(df, hdf5_filename, key=h5_key, mode=("w" if first else "a"))
            first = False
    except Exception as e:
        io_logger.error(f"HDF5 bundle write failed: {e}", exc_info=True)
        return ("Error","Error") if return_hash else None

    io_logger.info(f"HDF5 bundle write OK: {hdf5_filename}")

    if return_hash:
        try:
            md5, sha256 = calculate_hashes(hdf5_filename)
            return md5, sha256
        except Exception as e:
            io_logger.error(f"HDF5 hash failed: {e}", exc_info=True)
            return "Error", "Error"

    return None

# ----------------------------------
def save_to_parquet_bundle(
    dfs: dict,
    parquet_dir: str,
    base_filename_suffix: str,
    *,
    return_hash: bool = False,
    include: list | None = None,
):
    os.makedirs(parquet_dir, exist_ok=True)

    keys = DEFAULT_TABLE_ORDER if include is None else list(include)

    md5_map = {k: "N/A" for k in keys}
    sha_map = {k: "N/A" for k in keys}

    for k in keys:
        df = dfs.get(k)
        if df is None:
            continue

        stem = k.replace("PW", "PR").replace("QW", "QR")
        fn = os.path.join(parquet_dir, f"{base_filename_suffix}_{stem}.parquet")
        try:
            save_to_parquet(df, fn)
        except Exception as e:
            io_logger.error(f"Parquet write failed for {k}: {e}", exc_info=True)
            md5_map[k] = "Error"
            sha_map[k] = "Error"
            continue

        if return_hash:
            try:
                md5, sha256 = calculate_hashes(fn)
                md5_map[k] = md5
                sha_map[k] = sha256
            except Exception as e:
                io_logger.error(f"Parquet hash failed for {k}: {e}", exc_info=True)
                md5_map[k] = "Error"
                sha_map[k] = "Error"

    io_logger.info(f"Parquet bundle write OK: dir={parquet_dir}")

    if return_hash:
        return md5_map, sha_map
    return None


def save_to_csv_head_bundle(
    dfs: dict,
    out_dir: str,
    base_filename_suffix: str,
    *,
    csv_head_rows: int = 10,
    include: list | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    keys = DEFAULT_TABLE_ORDER if include is None else list(include)
    try:
        for k in keys:
            df = dfs.get(k)
            if df is None:
                continue
            stem = k.replace("PW", "PR").replace("QW", "QR")
            fn = os.path.join(out_dir, f"{base_filename_suffix}_{stem}.csv")
            save_to_csv(df.head(csv_head_rows), fn)

        io_logger.info(f"CSV head bundle write OK: dir={out_dir}, rows={csv_head_rows}")
    except Exception as e:
        io_logger.error(f"CSV head bundle write failed for {k}: {e}", exc_info=True)
# ----------------------------------
#MAIN interface
def save_all_bundles(
    dfs: dict,
    *,
    base_filename_suffix: str,
    sql_dir: str,
    hdf5_dir: str,
    parquet_dir: str,
    csv_dir: str,
    csv_head_rows: int = 10,
    return_hash: bool = True,
    include: list | None = None,
):
    os.makedirs(sql_dir, exist_ok=True)
    os.makedirs(hdf5_dir, exist_ok=True)
    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # SQLite
    db_filename = os.path.join(sql_dir, base_filename_suffix + ".db")
    db_md5, db_sha256 = save_to_db(dfs=dfs, db_path=db_filename, return_hash=return_hash) if return_hash else (None, None)

    # HDF5
    hdf5_filename = os.path.join(hdf5_dir, base_filename_suffix + "_combined.h5")
    hdf5_md5, hdf5_sha256 = save_to_hdf5_bundle(dfs, hdf5_filename, return_hash=return_hash, include=include) if return_hash else (None, None)

    # Parquet
    parq = save_to_parquet_bundle(dfs, parquet_dir, base_filename_suffix, return_hash=return_hash, include=include)
    if return_hash:
        parquet_md5_map, parquet_sha_map = parq
    else:
        parquet_md5_map, parquet_sha_map = {}, {}

    # CSV head (debug only)
    if csv_head_rows>0: save_to_csv_head_bundle(dfs, csv_dir, base_filename_suffix, csv_head_rows=csv_head_rows, include=include)

    return {
        "paths": {
            "db": db_filename,
            "hdf5": hdf5_filename,
            "parquet_dir": parquet_dir,
            "csv_dir": csv_dir,
        },
        "hash": {
            "db_md5": db_md5, "db_sha256": db_sha256,
            "hdf5_md5": hdf5_md5, "hdf5_sha256": hdf5_sha256,
            "parquet_md5": parquet_md5_map,
            "parquet_sha256": parquet_sha_map,
        }
    }
