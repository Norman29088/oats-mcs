#==================================================================
# run_oats_mcs.py
# OATS-MCS Sampling Entrypoint
# Standalone execution script for scenario generation
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
# Last updated: 2026-02
#==================================================================

from __future__ import division
import time, os,  pandas as pd, numpy as np, datetime, logging, uuid, threading
from tqdm import tqdm, trange
import polars as pl 
from typing import Tuple, Optional, Any
from pyomo.environ import Suffix
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from logging.handlers import RotatingFileHandler
import importlib.util,sys,random,gc
from oats_mcs.io_mcs_storage import save_all_bundles
from oats_mcs.io_mcs_orders import (prepare_scenario_inputs,create_data_order,merge_pf_LT,
                                    read_and_validate_orders,align_round_num_and_truncate)
from oats_mcs.io_mcs_dataset import build_row_from_spec,OatsMcsDataset
import gc, csv
from oats_sc.selecttestcase_sc import selecttestcase_sc 
from oats_sc.printdata_sc import printdata_sc 
from pyomo.environ import value
from pyomo.core.base.var import VarData
import threading
thread_local = threading.local()

#----------------------------------------------------------------------------------------------------------------------
RESET = "\033[0m" 
TASK_COLORS = {}  

def color_for_task(task_id):
    color_id = hash(task_id) % 256
    return f"\033[38;5;{color_id}m"

def color_print(msg, task_id):
    if task_id not in TASK_COLORS:
        TASK_COLORS[task_id] = color_for_task(task_id)
    color = TASK_COLORS[task_id]
    print(color + msg + RESET)

#----------------------------------------------------------------------------------------------------------------------
def progress_bar(current, total, start_time, bar_length=30, color_str="\033[37m",unique_id=""):
    elapsed = time.perf_counter() - start_time
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    avg_time = elapsed / current if current > 0 else 0
    remaining = avg_time * (total - current)
    msg = f"\r{unique_id} Progress: [{bar}] {percent*100:6.2f}% | ETA: {remaining:5.1f}s"
    sys.stdout.write(color_str + msg + RESET)
    sys.stdout.flush()

#----------------------------------------------------------------------------------------------------------------------

oats_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'OATS_log')
os.makedirs(oats_log_dir, exist_ok=True)
unique_suffix = uuid.uuid4().hex[:9]
oats_log_file = os.path.join(oats_log_dir, f'OATSoatslog_{os.getpid()}_{threading.get_ident()}_{unique_suffix}.log')

oats_logger = logging.getLogger()  
oats_logger.setLevel(logging.INFO)
for h in oats_logger.handlers[:]:
    oats_logger.removeHandler(h)

handler = RotatingFileHandler(oats_log_file, maxBytes=10*1024*1024, backupCount=10000, encoding="utf-8")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
handler.setFormatter(formatter)
oats_logger.addHandler(handler)
oats_logger.propagate = False

#----------------------------------------------------------------------------------------------------------------------
oats_dir = os.path.dirname(os.path.realpath(__file__))
default_testcase = os.path.join(oats_dir, 'testcases', 'IEEE-24R/IEEE-24R.xlsx')
#----------------------------------------------------------------------------------------------------------------------

def dcopfml(
    tc: str = "default",
    solver: str = "gurobi",
    model: str = "DCOPF",
    verbose: int = 0,
    num_rounds: int = 10,
    dem_matrix: Optional[Any] = None,   # PD: ndarray / pandas / polars
    res_matrix: Optional[Any] = None,   # PRES: optional
    excel_out: bool = False,
    rand_seed: Optional[int] = None,
    current_time: str = None,
    seed_num: Optional[int] = None,     # logging id
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Monte Carlo simulation for OPF/SCOPF models (DC), driven by scenario matrices.
    Parameters:
      tc: testcase workbook path.
      solver/model: solver and model controls.
      num_rounds: scenario count (must match the truncated/aligned matrices).
      dem_matrix/res_matrix: active power scenario matrices (PD/PRES).
      excel_out: whether to write Excel outputs.
      rand_seed: RNG seed for this run.
      current_time: timestamp string for file naming/logging.
      seed_num: logging identifier (e.g., same as rand_seed).

    Returns:
      dfs: result tables (implementation-dependent, typically pG/PD/PF/(QG/QD/...) dataframes).
      timing: timing summary dataframe (solver/internal/python timings).
    """

    if tc == 'default':tc = default_testcase

    # Solver / MCS options passed to runcase / solver
    opt = {
    "solver": solver,              # Solver name (e.g., gurobi, ipopt)
    "verbose": verbose,            # Verbosity level
    "round": num_rounds,            # Scenario count (aligned & truncated)
    "rand_seed": rand_seed,         # RNG seed for this run
    "dem_matrix": dem_matrix,       # PD: demand active power scenarios
    "res_matrix": res_matrix,       # PRES: RES active power scenarios (optional)
    "Excel_out": excel_out,         # Whether to write Excel outputs
    }
    testcase = tc
    mod = model.upper()
    logging.info(f"Solver selected: {solver}")
    logging.info(f"Testcase selected: {testcase}")
    logging.info(f"Model selected: {model}")
    logging.info(f"Round selected: {num_rounds}")
    logging.info(f"Rand_seed: {rand_seed}\n")
    print(f"\n{current_time} dcopfml model selected: {mod}\n")
    dfs, timing = runcase(testcase, mod, opt,current_time=current_time, seed_num=seed_num)
    logging.info("dcopfml done!")
    return dfs, timing 

def load_model(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def get_instance_value(inst_obj):
    if isinstance(inst_obj, VarData):
        return inst_obj.value
    else:
        return value(inst_obj)

def cleanup_case(instance=None, results=None, r=None, err =0, scen =0):
    try:
        if instance is not None:
            try:
                instance.solutions.clear()
            except Exception:
                pass
            try:# dual suffix can be large; drop it
                if instance is not None and hasattr(instance, "dual"):
                    instance.dual = None #instance.del_component(instance.dual)
            except Exception:
                pass
    except Exception:
        pass

    try:
        del instance, results, r
    except:
        pass

    if err > 0 and (err % 10 == 0):
        gc.collect()
    elif scen % 50 == 0:
        gc.collect()

    return None, None, None

#----------------------------------------------------------------------------------------------------------------------
def runcase(testcase, mod, opt=None,desired_order=None,current_time=None,seed_num=None):
    """Handling multiple scenarios, returning (pG_results, PD_results, PF_results, solver_time_total)."""
    if mod != "SCOPF_PREVENTIVE": raise ValueError("This branch only supports SCOPF_PREVENTIVE.")
    oats_dir = os.path.join(os.getcwd(), "oats_sc") 
    if isinstance(seed_num, (list, tuple)):
        assert len(seed_num) == 1, "seed_num list must have exactly one element"
        seed_num = seed_num[0]
    seed_num = 999999 if seed_num is None else int(seed_num)
    
    try:
        file_path = os.path.join(oats_dir, mod + '.py')
        modelf = load_model(mod, file_path)
        model = modelf.model
        logging.info("Given model file found in the models library")
    except Exception as e:
        logging.error("Given model file not found", exc_info=True)
        print("Failed to load the model, please check if the model file exists in:\n", file_path)  

        
    try:
        ptc = selecttestcase_sc(testcase) #RES-SCOPF
        logging.info("Testcase loaded")
    except Exception as e:
        logging.error("Testcase not found", exc_info=True)


    range_num = opt['round']
    dem_matrix = opt.get('dem_matrix', None)
    res_matrix  = opt.get("res_matrix", None)    
    excel_out = opt.get('Excel_out', False)
    datfile = f"datafile_{uuid.uuid4().hex}.dat"
    thread_local.logger.info(f"Created datfile {datfile}.")

    scenario_start_time  = time.perf_counter()
    solver_wall_time_internal =0.0
    solver_cpu_time_total = 0.0
    instance_wall_time_total = 0.0
    instance_full_time_total  = 0.0

    meta_rows = [] 

    try:
        order = create_data_order(model=mod, testcase=testcase)   # no external logger
        dataset = OatsMcsDataset(order, logger=None, ) 

        logging.info(f"[dataset:init] ok | model={mod} | tables={list(order.tables.keys())} | testcase={testcase}")
        thread_local.logger.info(f"[dataset:init] ok | model={mod} | tables={list(order.tables.keys())}")
    except Exception as e:
        logging.error(f"[dataset:init] failed | model={mod} | testcase={testcase} | err={e}", exc_info=True)
        thread_local.logger.error(f"[dataset:init] failed | err={e}", exc_info=True)
        print(f"----- Dataset initialization error, abort run | err={e} -----")

    err_flag = 0
    scenario_id = None

    for scen in range(range_num):
        instance = results = r = None 
        thread_local.logger.info(f"Start Solving Scenario {scen}")
        try: 
            dem_list,  res_list= prepare_scenario_inputs(scen,dem_matrix=dem_matrix,res_matrix=res_matrix,)

            if mod == "SCOPF_PREVENTIVE":
                r = printdata_sc(datfile, ptc, mod, opt,dem_list,  res_list)

            r.printheader();r.reducedata()                              
            r.printkeysets(); r.printnetwork(); r.printOPF(); r.printDC(); r.printSCdat()
            
            t00 = time.perf_counter() 
            solver_name = opt['solver']
            solve_engine = SolverFactory(solver_name)
            instance = model.create_instance(datfile)
            thread_local.logger.info(f"Scenario {scen} instance {datfile} created")

            instance.dual = Suffix(direction=Suffix.IMPORT)
            t0 = time.perf_counter()
            # results = solve_engine.solve(instance, tee=True, logfile="solverlog.txt")
            results = solve_engine.solve(instance, tee=False) 
            instance.solutions.load_from(results)

            t1 = time.perf_counter()

            # Potential CPU-time sources:
            #   1) results.solver.time (newer Pyomo; may not exist on macOS/Linux)
            #   2) results.solver.statistics['Time']
            solver_time_attr = getattr(results.solver, 'time', None) # Solver-reported CPU time (multi-threaded sum; may not exist on macOS/Linux)
            solver_stats     = getattr(results.solver, 'statistics', {}) # stats dict

            solver_info = results.solver[0]
            solver_inner_time = float(solver_info['Wall time'])

            if 'Time' in solver_info:
                solver_cpu_time = float(solver_info['Time'])
            else:
                solver_cpu_time = 100.0

            # Inner solver wall-clock and CPU-time (highest priority) 
            if solver_info is None:
                solver_inner_time = 100.0
                if solver_time_attr is not None: 
                    solver_cpu_time = solver_time_attr
                elif isinstance(solver_stats, dict) and 'Time' in solver_stats: 
                    solver_cpu_time = float(solver_stats['Time'])
                else:
                    solver_cpu_time = 100.0

            # Accumulate across solves
            solver_wall_time_internal += solver_inner_time
            solver_cpu_time_total += solver_cpu_time

            # Python-level wall-clock for this solve call
            solver_wall_time = t1 - t0
            instance_wall_time_total += solver_wall_time
            instance_full_time = t1 - t00
            instance_full_time_total += instance_full_time 

            scenario_id = f"s{int(seed_num):06d}_{int(scen):06d}"
            meta_rows.append({"scenario": scenario_id,"solver_status": str(results.solver.status),"term_cond":   str(results.solver.termination_condition),})


            # Progress Bar
            progress_interval = max(10, int(range_num*0.01)) 
            if (scen + 1) % progress_interval == 0:
                progress_bar(scen + 1, range_num, scenario_start_time ,unique_id=current_time)
                print()  

            dataset.append_from_instance(instance, scenario_id,results=results)
            thread_local.logger.info(f"Scenario {scen} collected: {scenario_id}")
        except Exception as e:
            thread_local.logger.error(f"Scenario {scen} collection failed: {scenario_id} | {e}", exc_info=True)
            print(f"----- Scenario {scen} collection error, skip -----")
            err_flag += 1
        finally:
            instance, results, r = cleanup_case(instance, results, r, err_flag,scen)

    try: os.remove(datfile)
    except Exception as e: logging.error(f"Error removing datfile: {e}", exc_info=True)

    scenario_end_time = time.perf_counter()
    scenario_time = scenario_end_time - scenario_start_time
    logging.info(f"Finished {range_num} scenarios with collection in {scenario_time:.4f} seconds (overall).")
    logging.info(f"Solving {range_num} in solver_internal_time_total: {solver_wall_time_internal:.6f}s")
    logging.info(f"Solving {range_num} in solver_cpu_time_total:    {solver_cpu_time_total:.6f}s")
    logging.info(f"Solving {range_num} in instance_wall_time_total: {instance_wall_time_total:.6f}s")
    logging.info(f"Solving {range_num} in instance_full_time_total: {instance_full_time_total:.6f}s")

        
    # end: merge PF parts into a single PF table
    dfs = dataset.to_pandas()  # {"PG": df, "PD": df, "PF": df, ...}
    dfs = merge_pf_LT(dfs, keep_parts=True)
    print("\n\nFinish OATSMCS Sampling!\n\n")
    if excel_out and range_num > 1000: excel_out = False
    if excel_out:
        try:
            rand_seed = opt.get("rand_seed", seed_num)
            uuidd = uuid.uuid4().hex[:9]

            if current_time is not None:excel_out_path = f"OATS_MCS_result/{current_time}/Results_excel"
            else: excel_out_path = f"Results/Results_{uuidd}"
            os.makedirs(excel_out_path, exist_ok=True)

            out_name = f"{excel_out_path}/results_case{dem_matrix.shape[1]}_{range_num}r_seed{rand_seed}.xlsx"

            with pd.ExcelWriter(out_name) as writer:
                index_xlsx="scenario_id"
                from oats_mcs.io_mcs_storage import HDF5_KEY_MAP
                for k, s in HDF5_KEY_MAP.items():
                    if k in dfs and dfs[k] is not None:
                        # dfs[k].set_index(index_xlsx).to_excel(writer, sheet_name=s)
                        dfs[k].to_excel(writer, sheet_name=s,index=True)
            print(f"Results written to {out_name}")
            thread_local.logger.info(f"Excel write OK: {out_name}")
            logging.info(f"Excel write OK: {out_name}")

        except Exception as e:
            thread_local.logger.error(f"Excel write failed: {e}", exc_info=True)
            logging.error(f"Excel write failed: {e}", exc_info=True)
            print("Excel write error")

    #Pyomo interface-level reported cumulative CPU-time; Python-level time accumulation; solver's own internal wall-clock time accumulation; instantiation + solution process total time
    timing_info = {
        "solver_cpu_time_total": solver_cpu_time_total,
        "instance_wall_time_total": instance_wall_time_total,
        "solver_wall_time_internal": solver_wall_time_internal,
        "instance_full_time_total": instance_full_time_total,
    }

    return dfs, timing_info



#------------------------------------------------------
# Seed-level independent logger
def get_seed_logger(seed, oats_MCS_path, current_time,log_level=logging.INFO):
    import os, threading
    logger  = logging.getLogger(f"customLogger_seed_{seed}_{os.getpid()}_{threading.get_ident()}")
    logger.handlers = []  
    logger.setLevel(log_level)
    log_file = os.path.join(oats_MCS_path, f"OATS_MCS_log_{seed}_{current_time}.log")
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=10, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False 
    logger.info(f"OATS_MCS Start: for seed {seed}")
    thread_local.logger = logger
    return logger 
#------------------------------------------------------

#------------------------------------------------------
def run_oats_mcs(
    solver="gurobi",
    model="SCOPF_PREVENTIVE",
    round_num=10,                  # Scenario count
    num_generators=33,              # Number of PG variables (2-D width)
    num_demands=17,                 # Number of PD variables (2-D width)
    num_res=0,                      # Number of RES variables (2-D width); 
    res_sampling=True,             #Whether to sampling RES
    testcase="OATS-testcases/case24_ieee_rts.xlsx",
    seed_nums=[312324],             # List of RNG seeds for repeated runs

    excel_out=False,                 # Write Excel results in runcase (optional)
    save_to_sql=True,               # Export generated scenario matrices to SQLite (debug/inspection)
    save_to_csv=True,               # Export generated scenario matrices to CSV (debug/inspection)

    # -------------------------
    # Demand (PD) scenario generation
    expect_range_dem=(0.55, 1.55),  # PD scaling range (per-variable)
    demand_dist="uniform",          # PD sampling distribution

    # -------------------------
    # RES (PRES) scenario generation
    expect_range_res=(0.55, 1.55),  # RES scaling range
    res_dist="uniform",             # RES sampling distribution
):
    """
    Run Monte Carlo Simulation (MCS) for Preventive DC-SCOPF dataset generation.
    with configurable scenario generation for demand / generation / RES and optional
    1-D random perturbations.

    Parameters:
        solver (str): Solver name, e.g., "gurobi" or "ipopt".
        model (str): Power-system optimisation model, e.g., DCOPF / ACOPF / SCOPF.
        round_num (int): Number of scenarios to generate and solve.
        num_generators (int): Number of  variables (2-D scenario width).
        num_demands (int): Number of demand variables (2-D scenario width).
        num_res (int): Number of RES variables (2-D scenario width).
        testcase (str): Path to the testcase workbook (baseline network data).
        resultcase (str | None): Reserved/unused output workbook path.
        seed_nums (list[int]): List of random seeds for multiple runs.
        excel_out (bool): Whether to write Excel results (handled in runcase).
        save_to_sql (bool): Whether to export generated scenario tables to SQLite.
        save_to_csv (bool): Whether to export generated scenario tables to CSV.
        expect_range_dem (tuple[float, float]): Demand scaling range (low, high).
        demand_dist (str): Demand sampling distribution name (e.g., "uniform", "normal", "pareto").
        expect_range_res (tuple[float, float]): RES scaling range (low, high).
        res_dist (str): RES sampling distribution name.
        random_low (float): Lower bound of 1-D random perturbation (e.g., for bounds slack).
        random_high (float): Upper bound of 1-D random perturbation.
        verbose (int): Verbosity level.

    Returns:
        Dict[str, Any]: Run metadata and aggregated results (implementation-dependent).

    """
    try:
        logging.info(f"Before cleaning, solver: '{solver}', model: '{model}'")
        solver = str(solver).strip().lower()  # Normalize solver to lowercase
        model = str(model).strip().upper()  # Normalize model to uppercase
        logging.info(f"After cleaning, solver: '{solver}', model: '{model}'")
        

        try:
            num_generators, num_demands, num_res = read_and_validate_orders(
                testcase=testcase,
                num_generators=num_generators,
                num_demands=num_demands,
                num_res=num_res,
            )
        except Exception:
            logging.error("Failed to read or validate order from testcase", exc_info=True)

        # Log storage directory (log name generated based on the current time, including millisecond-level timestamps + UUID)
        try:
            import uuid
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_f')[:-3]+ "_Task"+"_" + uuid.uuid4().hex[:9]
            oats_MCS_path = f"OATS_MCS_result/{current_time}"
            os.makedirs(oats_MCS_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating result directory: {e}", exc_info=True)
        
        start_total = time.perf_counter()
        end_total = time.perf_counter() #Assign values ​​in advance to prevent interruption and error return.
        results = []  # # Store the results for each seed
        for seed_num in seed_nums:
            try:
                logger = get_seed_logger(seed_num, oats_MCS_path, current_time, log_level=logging.ERROR)
                if logger is None or not logger.handlers:
                    logging.warning(f"Logger for seed {seed_num} was not set up correctly. Skipping this seed.")
                    continue
                
                print(f"\n=== Start seed = {seed_num} ===")
                # # 2) Generate generator / demand / res random matrix (two-dimensional) 
                from oats_mcs.data_generator import get_dataset
                dataset_common_args = {
                        'num_scen': round_num,
                        "dtype": np.float64,
                        "write_to_sql": save_to_sql,
                        "save_to_excel": save_to_csv,
                        "out_sql_table": None,
                        'baseline_path': testcase,
                        'out_sql_path':f"{oats_MCS_path}/Dataset_generated_{seed_num}_{round_num}R_{current_time}.db",
                        'rand_seed': seed_num,
                    }
                try:
                    config_dem = {
                        'num_variable': num_demands,
                        'expect_range': expect_range_dem, 
                        'distribution': demand_dist,      
                        'out_excel_path': f"{oats_MCS_path}/Dem_generated_{seed_num}_{round_num}R_{current_time}.csv",
                        'baseline_col': "real", 
                        **dataset_common_args,
                    }
                    dem_matrix = get_dataset("demand", **config_dem)                    
                    print("Demand matrix shape:", dem_matrix.shape,f"\nDemand use rand_seed {seed_num}")
                except Exception as e:
                    logger.error(f"Error generating demand matrix: {e}", exc_info=True)
                    continue
                res_matrix = None

                # RES / Wind (or RES)
                if res_sampling:
                    try:
                        config_res = {
                            "num_variable": num_res,
                            "expect_range": expect_range_res,
                            "distribution": res_dist,
                            "out_excel_path": f"{oats_MCS_path}/Res_generated_{seed_num}_{round_num}R_{current_time}.csv",
                            "baseline_col": "PGUB", 
                            **dataset_common_args,
                        }
                        res_matrix = get_dataset("wind", **config_res)
                        print("RES matrix shape:", res_matrix.shape,f"\nRES use rand_seed {seed_num+ 25000}")
                    except Exception as e:
                        logger.error(f"Error generating PG/RES matrix: {e}", exc_info=True);res_matrix=None
                        continue
                else: res_matrix=None

                
                round_num, dem_matrix, res_matrix, = \
                    align_round_num_and_truncate(
                        seed_num=seed_num,
                        round_num=round_num,
                        model=model,
                        dem_matrix=dem_matrix,
                        res_matrix=res_matrix,
                        logger=logging,
                    )

                try:
                    dfs, timing = dcopfml(
                        tc=testcase,
                        solver=solver,
                        model=model,
                        verbose=0,
                        num_rounds=round_num,   
                        dem_matrix=dem_matrix,    
                        res_matrix=res_matrix,     
                        excel_out=excel_out,
                        rand_seed=seed_num,
                        current_time=current_time,
                        seed_num=seed_num 
                    )
                    solver_cpu_time_total       = timing["solver_cpu_time_total"]
                    instance_wall_time_total    = timing["instance_wall_time_total"]
                    solver_wall_time_internal   = timing["solver_wall_time_internal"]
                    instance_full_time_total    = timing["instance_full_time_total"]
                except Exception as e:
                    logger.error(f"Error running dcopfml: {e}", exc_info=True)
                    continue

                summary = save_all_bundles(
                    dfs,
                    base_filename_suffix = f"oats_scens_case{num_demands}_{round_num}r_seed{seed_num}",
                    sql_dir = os.path.join("OATS_MCS_result", "sql_files"),
                    hdf5_dir = os.path.join("OATS_MCS_result", "hdf5_files"),
                    parquet_dir = os.path.join("OATS_MCS_result", "parquet_files"),
                    csv_dir = os.path.join("OATS_MCS_result", "csv_files"),
                    csv_head_rows=0, 
                    return_hash=True,
                )

                db_md5 = summary["hash"]["db_md5"]
                db_sha256 = summary["hash"]["db_sha256"]
                hdf5_md5 = summary["hash"]["hdf5_md5"]
                hdf5_sha256 = summary["hash"]["hdf5_sha256"]
                parquet_md5_map = summary["hash"]["parquet_md5"]
                parquet_sha_map = summary["hash"]["parquet_sha256"]

                try:
                    del dfs
                    gc.collect()
                    thread_local.logger.info("DataFrames deleted and garbage collected successfully.")
                except Exception as e:
                    thread_local.logger.error(f"Error deleting DataFrames or garbage collecting: {e}", exc_info=True)
                end_total = time.perf_counter()
                total_runtime = end_total-start_total
                print('Total operation time =', total_runtime)

            except Exception as e:
                logging.exception("Unhandled exception in seed loop:")
                continue 
        return {
            "Case_number": current_time,
            "actual_scen_num": round_num,
            "actual_dem_num": num_demands,
            "actual_gen_num": num_generators,
            "solver_wall_time_internal": solver_wall_time_internal,  # Cumulative internal solver wall-clock time 
            "solver_cpu_time_total": solver_cpu_time_total,          # Cumulative CPU time at the Pyomo interface level 
            "instance_wall_time_total": instance_wall_time_total,    # Cumulative Python-level wall-clock time 
            "instance_full_time_total": instance_full_time_total,    # Cumulative end-to-end time (instantiation + solve) 
            "MCS_runningtime": total_runtime,
            "db_md5": db_md5,
            "db_sha256": db_sha256,
            "hdf5_md5": hdf5_md5,
            "hdf5_sha256": hdf5_sha256,
            "parquet_md5": parquet_md5_map, 
            "parquet_sha256": parquet_sha_map,
        }
    except: 
        logging.exception("Unhandled exception in run_oats_mcs:")




if __name__ == "__main__":
    run_oats_mcs()


