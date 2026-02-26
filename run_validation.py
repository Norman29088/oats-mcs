#==================================================================
# run_validation.py
# OATS-MCS Preventive SCOPF Validation (user entry script)
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
# Last updated: 2026-02
#==================================================================



from oats_val.oatsmcs_validation import load_and_validate_dataset
import torch
from pathlib import Path
import pandas as pd
import sqlite3 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using {device}")



# batch_size:
#   None  -> full read/write (load entire table into memory)
#   int   -> batch mode (process in chunks of size=batch_size)
batch_size = None
batch_size = 100000

# TODO: INPUT datasets (SQLite files to validate)
db_list = [
    # "IEEE24R-A/IEEE24R-A.db",
    "IEEE24_demo_30r/IEEE24_demo_30r.db",
]

# Check tables
for db in db_list:
    with sqlite3.connect(db) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )]
    print(db, tables)


base_config = dict(
    testcase="IEEE-24R/IEEE-24R.xlsx",
    batch_size=batch_size,
    resultcase="IEEE-24R/results.xlsx",
    grid_param_path="IEEE-24R/N-1_contingency_test",
    atol=1e-5,
    rtol=1e-5,
    device=device,
    dtype=torch.float64,
)



for db_path in db_list:

    dc_rows, sc_rows = [], []
    dc_vio_rows, sc_vio_rows = [], []
    gen_vio_rows, pw_vio_rows = [], []

    PTDF_SC_full, dc_report, sc_report, dc_vio_report, sc_vio_report, oats_dict, gen_vio_report, pw_vio_report = \
        load_and_validate_dataset(**base_config, db_path=db_path)

    stem = Path(db_path).stem

    for r in dc_report:
        dc_rows.append(dict(db=stem, **r))
    for r in sc_report:
        sc_rows.append(dict(db=stem, **r))
    for r in dc_vio_report:
        dc_vio_rows.append(dict(db=stem, **r))
    for r in sc_vio_report:
        sc_vio_rows.append(dict(db=stem, **r))
    for r in gen_vio_report:
        gen_vio_rows.append(dict(db=stem, **r))
    for r in pw_vio_report:
        pw_vio_rows.append(dict(db=stem, **r))

    df_dc = pd.DataFrame(dc_rows)
    df_sc = pd.DataFrame(sc_rows)
    df_dc_vio = pd.DataFrame(dc_vio_rows)
    df_sc_vio = pd.DataFrame(sc_vio_rows)
    df_gen_vio = pd.DataFrame(gen_vio_rows)
    df_pw_vio = pd.DataFrame(pw_vio_rows)

    out_dir = Path("validation_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{stem}_report.xlsx"

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_dc.to_excel(writer, sheet_name="DC_Report", index=False)
        df_sc.to_excel(writer, sheet_name="SC_Report", index=False)
        df_dc_vio.to_excel(writer, sheet_name="DC_Violation", index=False)
        df_sc_vio.to_excel(writer, sheet_name="SC_Violation", index=False)
        df_gen_vio.to_excel(writer, sheet_name="GEN_Violation", index=False)
        df_pw_vio.to_excel(writer, sheet_name="PW_Violation", index=False)

    print(f"[Saved] {out_path}")




