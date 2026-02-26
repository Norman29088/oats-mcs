#==================================================================
# run_oatsmcs.py
# OATS-MCS Preventive SCOPF Sampling (user entry script)
# Author: Runsheng He
# Copyright (c) 2026 Runsheng He
# Licensed under the GNU General Public License v3.0 (GPLv3)
# Last updated: 2026-02
#==================================================================
from oats_mcs.run import run_oats_mcs


if __name__ == "__main__":

    testcase= 'IEEE-24R/IEEE-24R.xlsx'
    
    result_dict = run_oats_mcs(
        solver="gurobi",
        model="SCOPF_PREVENTIVE",
        round_num=30,
        res_sampling=True,
        testcase=testcase,
        seed_nums=[33000],  # For uniform/lhs/sobol: rand_seed = seed_main*1000 + block_id (last 3 digits = block index)
        excel_out=True,
        save_to_sql=True,
        save_to_csv=True,
        expect_range_dem=(0.5, 1.5),
        demand_dist="lhs_30",
        expect_range_res=(0.0, 1),
        res_dist="multi_uniform",         # uniform / lhs_<N> / sobol /multi_uniform / multi_lhs; N:Total samples; eg: LHS_10000
    )
    ## Sequential sampling is allowed
    # result_dict = run_oats_mcs(
    #     solver="gurobi",
    #     model="SCOPF_PREVENTIVE",
    #     round_num=5,
    #     res_sampling=True,
    #     testcase=testcase,
    #     seed_nums=[33001],
    #     excel_out=True,
    #     save_to_sql=True,
    #     save_to_csv=True,
    #     expect_range_dem=(0.5, 1.5),
    #     demand_dist="lhs_15",
    #     expect_range_res=(0.05, 1),
    #     res_dist="uniform",        
    # )
    # # print(result_dict)