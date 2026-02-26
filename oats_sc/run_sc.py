#==================================================================
# run_sc.py
# Top-level OATS script for single-scenario simulation.
#
# --- Original Author ---
# W. Bukhsh (OATS)
# Copyright (c) 2017 W. Bukhsh
# Licensed under the GNU General Public License v3.0 (GPLv3).
#
# --- Modifications ---
# Copyright (c) 2026 R. He
# Project: OATS-MCS (Monte Carlo Sampling extension for OATS)
# Adapted for single-scenario preventive DC-SCOPF execution.
# Date: Feb 2026
#==================================================================
from __future__ import division
import logging
import os
import imp 
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import logging
from oats_sc.selecttestcase_sc import selecttestcase_sc
from oats_sc.printdata_sc import printdata_sc 
from oats_sc.printoutput_sc import printoutput
import importlib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='oatslog.log',
                    filemode='w')
logging.info("OATS log file")
logging.info("Program started")
oats_dir = os.path.dirname(os.path.realpath(__file__))
default_testcase = oats_dir+'/testcases/case24_ieee_rts.xlsx'


# Preventive security constrained optimal power flow problem
def SCOPF_PREVENTIVE(tc='default',solver='ipopt',out=0):
    """
    Solves security constrained optimal power flow problem
    ARGUMENTS:
        **tc** (*.xlsx file)  - OATS test case. See OATS data format for details
        **solver** (str)  - name of a solver. Defualt is 'ipopt'
        **out** (bool) - If True, the output is displayed on screen.
    """

    if tc == 'default':
        tc = default_testcase
    #options
    opt=({'solver':solver,'out':out})
    testcase = tc
    model ='SCOPF_PREVENTIVE'
    # ==log==
    logging.info("Solver selected: "+opt['solver'])
    logging.info("Testcase selected: "+testcase)
    logging.info("Model selected: "+model)
    runcase(testcase,model,opt)
    logging.info("Done!")

def load_model(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def runcase(testcase,mod,opt=None):
    oats_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(oats_dir, mod + '.py')
    modelf = load_model(mod, file_path)
    logging.info("Given model file found in the models library (user_def_model)")
    model = modelf.model
    try:
        ptc = selecttestcase_sc(testcase) 
        logging.info("Given testcase file found and selected from the testcase library")
    except Exception:
        logging.error("Given testcase  not found in the 'testcases' library", exc_info=False)
        raise
    datfile = 'datafile.dat'
    r = printdata_sc(datfile,ptc,mod,opt)
    r.reducedata()
    r.printheader()
    r.printkeysets()
    r.printnetwork()
    r.printOPF()
    r.printDC()
    r.printSCdat()

    optimise = SolverFactory(opt['solver'])

    instance = model.create_instance(datfile)
    instance.dual = Suffix(direction=Suffix.IMPORT)
    results = optimise.solve(
        instance,
        # opt=solveroptions,
        tee=False,           
        # tee=True,         
        # load_solutions=True  
    )
    instance.solutions.load_from(results)
    o = printoutput(results, instance,mod)
    if (opt['out']): o.solutionstatus()
    else:o.greet();o.solutionstatus()
    o.printsummary()
    o.printoutputxls()

    print ("DEbug Status:", results.solver.status) 
    print ("DEbug Termination condition:", results.solver.termination_condition)



