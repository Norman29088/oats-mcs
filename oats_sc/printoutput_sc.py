#==================================================================
# printout_sc.py
# A Python script to write output to xls and screen
#
# --- Original Author ---
# W. Bukhsh (OATS)
# Copyright (c) 2015 W. Bukhsh
# Licensed under the GNU General Public License v3.0 (GPLv3).
#
# --- Modifications ---
# Copyright (c) 2026 R. He
# Project: OATS-MCS (Monte Carlo Sampling extension for OATS)
# Minor adaptations for preventive DC-SCOPF workflow.
# Date: Feb 2026
#==================================================================
from pyomo.opt import SolverStatus, TerminationCondition
from tabulate import tabulate
import pandas as pd
import math
import sys
class printoutput(object):
    def __init__(self, results, instance,mod):
        self.results   = results
        self.instance  = instance
        self.mod       = mod
        self.baseMVA = self.instance.baseMVA.value
    def greet(self):
        print ("========================")
        print ("\n Output from the OATS")
        print ("========================")
    def solutionstatus(self):
        self.instance.solutions.load_from(self.results)
        print ("------Solver Message------")
        print (self.results.solver)
        print ("--------------------------")
        if (self.results.solver.status == SolverStatus.ok) \
        and (self.results.solver.termination_condition == TerminationCondition.optimal):
            print ("Optimization Converged!")
        elif self.results.solver.termination_condition == TerminationCondition.infeasible:
            sys.exit("Problem is infeasible!\nOats terminated. No output is written on the results file.")
        else:
            print (sys.exit("Problem is infeasible!\nOats terminated. No output is written on the results file."))
    def printsummary(self):
        if 'LF' not in self.mod:
            print ("Cost of the objective function:", str(float(self.instance.OBJ())))
        print ("***********")
        print ("\n Summary")
        print ("***********")
        tab_summary = []
        tab_summary.append(['Conventional generation (MW)','Renewable generation (MW)', 'Demand (MW)'])
        tab_summary.append([sum(self.instance.pG[g].value for g in self.instance.G)*self.baseMVA,\
        sum(self.instance.pW[w].value for w in self.instance.RES)*self.baseMVA,sum(self.instance.PD[d] for d in self.instance.D)*self.baseMVA])
        print (tabulate(tab_summary, headers="firstrow", tablefmt="grid"))
        print ("==============================================")
    def printoutputxls(self):
        #===initialise pandas dataframes
        cols_summary    = ['Conventional generation (MW)', 'Renewable generation (MW)', 'Demand (MW)','Objective function value']
        cols_bus        = ['name', 'angle(degs)']
        cols_demand     = ['name', 'busname', 'PD(MW)']

        if ('DC' in self.mod) or ('SC' in self.mod):
            cols_bus.append('LMP')
            cols_branch     = ['name', 'from_busname', 'to_busname', 'pL(MW)']
            cols_transf     = ['name', 'from_busname', 'to_busname', 'pLT(MW)']
            if 'LF' in self.mod:
                cols_generation = ['name', 'busname', 'PG(MW)', 'pG(MW)']
                cols_res       = ['name', 'busname', 'PG(MW)', 'pG(MW)']
            elif 'OPF' in self.mod:
                cols_branch.append('SLmax(MW)')
                cols_transf.append('SLmax(MW)')
                cols_generation = ['name', 'busname', 'PGLB(MW)','PG(MW)', 'pG(MW)','PGUB(MW)']
                cols_res       = ['name', 'busname', 'PGLB(MW)','PG(MW)', 'pG(MW)','PGUB(MW)']
        summary         = pd.DataFrame(columns=cols_summary)
        bus             = pd.DataFrame(columns=cols_bus)
        demand          = pd.DataFrame(columns=cols_demand)
        res            = pd.DataFrame(columns=cols_res)
        generation      = pd.DataFrame(columns=cols_generation)
        branch          = pd.DataFrame(columns=cols_branch)
        transformer     = pd.DataFrame(columns=cols_transf)

        #-----write Data Frames

        summary.loc[0] = pd.Series({'Conventional generation (MW)': sum(self.instance.pG[g].value for g in self.instance.G)*self.baseMVA,\
        'Renewable generation (MW)':sum(self.instance.pW[w].value for w in self.instance.RES)*self.baseMVA,\
        'Demand (MW)':sum(self.instance.PD[d] for d in self.instance.D)*self.baseMVA,\
        'Objective function value': self.instance.OBJ()})

        if ('DC' in self.mod) or ('SC' in self.mod):
            #bus data
            ind=0

            for b in self.instance.B:
                bus.loc[ind] = pd.Series({'name': b,'angle(degs)':self.instance.delta[b].value*180/math.pi,'LMP':self.instance.dual[self.instance.KCL_const[b]]/self.baseMVA})
                ind += 1

            if 'OPF' in self.mod:
                #line data
                ind=0
                for b in self.instance.L:
                    branch.loc[ind] = pd.Series({'name': b, 'from_busname':self.instance.A[b,1], 'to_busname':self.instance.A[b,2],\
                    'pL(MW)':self.instance.pL[b].value*self.baseMVA,'SLmax(MW)':self.instance.SLmax[b]*self.baseMVA})
                    ind += 1
                #transformer data
                ind = 0
                for b in self.instance.TRANSF:
                    transformer.loc[ind] = pd.Series({'name': b, 'from_busname':self.instance.AT[b,1],
                    'to_busname':self.instance.AT[b,2], 'pLT(MW)':self.instance.pLT[b].value*self.baseMVA,\
                    'SLmax(MW)':self.instance.SLmaxT[b]*self.baseMVA})
                    ind += 1

                #demand data
                ind = 0
                for d in self.instance.Dbs:
                    demand.loc[ind] = pd.Series({'name': d[1],'busname':d[0],'PD(MW)':self.instance.PD[d[1]]*self.baseMVA})
                    ind += 1
                #generator data
                ind = 0
                for g in self.instance.Gbs:
                    generation.loc[ind] = pd.Series({'name':g[1], 'busname':g[0],\
                    'PGLB(MW)':self.instance.PGmin[g[1]]*self.baseMVA,\
                    'pG(MW)':round(self.instance.pG[g[1]].value*self.baseMVA,3),\
                    'PGUB(MW)':self.instance.PGmax[g[1]]*self.baseMVA})
                    ind += 1

                #res data
                ind = 0
                for g in self.instance.Wbs:
                    res.loc[ind] = pd.Series({'name':g[1], 'busname':g[0],\
                    'PGLB(MW)':self.instance.WGmin[g[1]]*self.baseMVA,\
                    'PG(MW)':round(self.instance.WGmax[g[1]]*self.baseMVA,3),\
                    'pG(MW)':round(self.instance.pW[g[1]].value*self.baseMVA,3),\
                    'PGUB(MW)':self.instance.WGmax[g[1]]*self.baseMVA})
                    ind += 1
    
        #----------------------------------------------------------
        #===write output on xlsx file===
        bus = bus.sort_values(['name'])
        generation = generation.sort_values(['name'])
        demand = demand.sort_values(['name'])
        writer = pd.ExcelWriter('results.xlsx', engine ='xlsxwriter')
        summary.to_excel(writer, sheet_name = 'summary',index=False)
        bus.to_excel(writer, sheet_name = 'bus',index=False)
        demand.to_excel(writer, sheet_name = 'demand',index=False)
        generation.to_excel(writer, sheet_name = 'generator',index=False)
        res.to_excel(writer, sheet_name = 'renewable',index=False)
        branch.to_excel(writer, sheet_name = 'branch',index=False)
        transformer.to_excel(writer, sheet_name = 'transformer',index=False)
        writer.close()
    