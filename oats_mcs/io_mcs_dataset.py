#==================================================================
# io_mcs_dataset.py
# Defines the OATS-MCS dataset abstraction and loaders
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
#==================================================================


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal, Any
import logging
import pandas as pd
from pyomo.environ import value
from pyomo.core.base.var import VarData

def get_instance_value(inst_obj):
    if isinstance(inst_obj, VarData):
        return inst_obj.value
    else:
        return value(inst_obj)

def build_row_from_spec(instance, spec, scenario_id: str, resolver: Optional[Dict[str, object]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {"scenario_id": scenario_id}

    # baseMVA scaling factor
    base = 1.0
    if getattr(spec, "scale_baseMVA", True):
        base = float(get_instance_value(getattr(instance, "baseMVA")))

    container = getattr(instance, spec.src, None)
    if container is None:
        raise AttributeError(f"instance has no attribute '{spec.src}' for table {spec.name}")
    for k in spec.order:
        v = get_instance_value(container[k])
        row[k] = float(v) * base


    if getattr(spec, "extras", None):
        extras = set(spec.extras)

        if "probC" in extras:
            if hasattr(instance, "probC") and hasattr(instance, "C"):
                row["probC"] = float(sum(value(instance.probC[c]) for c in instance.C))
            else:
                row["probC"] = 0.0

        if "probOK" in extras:
            if hasattr(instance, "probOK"):
                row["probOK"] = float(get_instance_value(instance.probOK))
            else:
                row["probOK"] = 0.0

    return row

@dataclass
class OatsMcsDataset:
    data_order: Any  # DataOrder
    logger: Optional[logging.Logger] = None
    rows: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # per-instance resolver cache: {"G": {...}, "D": {...}, ...}
    _resolver_cache: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        # init row buffers for all planned tables
        for t in self.data_order.tables.keys():
            self.rows.setdefault(t, [])


    def append_from_instance(self, instance, scenario_id: str,results=None):
        order = self.data_order
        lg = self.logger

        built_rows: Dict[str, Dict[str, Any]] = {}

        for tname, spec in order.tables.items():
            row = build_row_from_spec(instance, spec, scenario_id, resolver=None)
            built_rows[tname] = row

        if "PD" in built_rows and "PG" in built_rows:
            pd_row = built_rows["PD"]
            pg_row = built_rows["PG"]

            # sum only numeric asset columns (exclude meta)
            def _sum_row(r: dict) -> float:
                keys = [k for k in r.keys() if k not in ("scenario_id",)]
                return float(sum(r[k] for k in keys if isinstance(r.get(k), (int, float))))

            pd_sum = _sum_row(pd_row)
            pg_sum = _sum_row(pg_row)

            pw_sum = 0.0
            if "PW" in built_rows:
                pw_row = built_rows["PW"]
                pw_sum = _sum_row(pw_row)
                # write total into PW row itself
                pw_row["PR_Total"] = pw_sum

            # write totals into PD/PG rows themselves
            pd_row["PD_Total"] = pd_sum
            pg_row["PG_Total"] = pg_sum
            pg_row["PR_Total"] = pw_sum
            pg_row["PD_Total"] = pd_sum

            # P balance (no loss / no VOLL): PG + PW - PD
            p_diff = (pg_sum + pw_sum) - pd_sum
            pg_row["P_Diff"] = p_diff

            pg_row["solver_status"] = str(results.solver.status)
            pg_row["term_cond"]     = str(results.solver.termination_condition)

        for tname, row in built_rows.items():
            self.rows[tname].append(row)

    def to_pandas(self) -> Dict[str, pd.DataFrame]:
        out = {}
        for tname, lst in self.rows.items():
            out[tname] = pd.DataFrame(lst)
        return out

