#==================================================================
# run_nn_demo.py
# Standalone execution script for neural network training demo
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
# Last updated: 2026-02
#==================================================================

from oats_ml.oatsmcs_training import train_scopf_mlp
import torch
from pathlib import Path



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] {device}")

base_config = dict(
    testcase="IEEE-24R/IEEE-24R.xlsx",
    resultcase="IEEE-24R/results.xlsx",
    grid_param_path="IEEE-24R/N-1_contingency_test",
    batch_size=None,
    # batch_size=3,
    split_ratio=(8, 1, 1),
    atol=1e-5,
    rtol=1e-5,
    device=device,
    dtype=torch.float64,
    random_seed=33000,
    hidden_dim=32,
    n_hidden=3,
    epochs=100,
    lr=5e-3,
    use_clip=True,
    use_bn=True,
)

db_list = [
    # "IEEE24R-A/IEEE24R-A.db",
    "IEEE24_demo_30r/IEEE24_demo_30r.db",
]

for db_path in db_list:

    db_path = Path(db_path).resolve()
    stem = db_path.stem
    print(f"\n==== Train: {stem} ====")

    model = train_scopf_mlp(db_path=str(db_path), **base_config)

    run_dir = db_path.parent / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "mlp.pt"

    torch.save(model.state_dict(), model_path)
    print(f"[Saved] {model_path}")