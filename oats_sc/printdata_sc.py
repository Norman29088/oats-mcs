#==================================================================
# printdata_sc.py
# A Python script to write data file for PYOMO
#
# --- Original Author ---
# W. Bukhsh (OATS)
# Copyright (c) 2015 W. Bukhsh
# Licensed under the GNU General Public License v3.0 (GPLv3).
#
# --- Modifications ---
# Copyright (c) 2026 R. He
# Project: OATS-MCS (Monte Carlo Sampling extension for OATS)
# Adaptations: Preventive DC-SCOPF workflow integration.
# Date: Feb 2026
#==================================================================
import datetime
import math
import sys
deltaT = 1.0
class printdata_sc(object):
    def __init__(self,datfile,data,model,options,dem_list=None,res_list=None,):
        self.datfile = datfile
        self.data    = data
        self.model   = model
        self.options = options
        self.dem_list = dem_list     # PD scenario (row)
        self.res_list = res_list     # RES(P)/WIND scenario (row)
        if self.dem_list is not None:
            self.data["demand"]["real"] = self.dem_list  #DC-PD
        if self.res_list is not None:
            self.data["res"]["PGUB"] = self.res_list  
    def reducedata(self):
        self.data["demand"]      = self.data["demand"].drop(self.data["demand"][self.data["demand"]['stat'] == 0].index.tolist())
        self.data["branch"]      = self.data["branch"].drop(self.data["branch"][self.data["branch"]['stat'] == 0].index.tolist())
        if self.data["flags"]["shunt"]:
            self.data["shunt"]       = self.data["shunt"].drop(self.data["shunt"][self.data["shunt"]['stat'] == 0].index.tolist())
        if self.data["flags"]["storage"]:
            self.data["storage"]     = self.data["storage"].drop(self.data["storage"][self.data["storage"]['stat'] == 0].index.tolist())
        self.data["transformer"] = self.data["transformer"].drop(self.data["transformer"][self.data["transformer"]['stat'] == 0].index.tolist())
        self.data["res"]        = self.data["res"].drop(self.data["res"][self.data["res"]['stat'] == 0].index.tolist())
        self.data["generator"]   = self.data["generator"].drop(self.data["generator"][self.data["generator"]['stat'] == 0].index.tolist())

    def printheader(self):
        f = open(self.datfile, 'w')
        #####PRINT HEADER--START
        f.write('#This is Python generated data file for Pyomo model DCLF.py\n')
        f.write('#Original Author: W. Bukhsh\n')
        f.write('#Modified by: R. He (OATSMCS Project)\n')
        f.write('#Time stamp: '+ str(datetime.datetime.now())+'\n')
        f.close()

    def printkeysets(self):
        f = open(self.datfile, 'a')
        ##===sets===
        #---set of buses---
        f.write('set B:=\n')
        for i in self.data["bus"].index.tolist():
            f.write(str(self.data["bus"]["name"][i])+"\n")
        f.write(';\n')
        #---set of generators---
        f.write('set G:=\n')
        for i in self.data["generator"].index.tolist():
            f.write(str(self.data["generator"]["name"][i])+"\n")
        f.write(';\n')
        #---set of demands---
        f.write('set D:=\n')
        for i in self.data["demand"]["name"].unique():
            f.write(str(i)+"\n")
        f.write(';\n')
        #---set of Renewable generators---
        if not(self.data["res"].empty):
            f.write('set RES:=\n')
            for i in self.data["res"]["name"].unique():
                f.write(str(i)+"\n")
            f.write(';\n')
        #===parameters===
        #---Real power demand---
        f.write('param PD:=\n')
        for i in self.data["demand"].index.tolist():
            f.write(str(self.data["demand"]["name"][i])+" "+str(float(self.data["demand"]["real"][i])/self.data["baseMVA"]["baseMVA"][0])+"\n")
        f.write(';\n')
        f.write('param baseMVA:=\n')
        f.write(str(self.data["baseMVA"]["baseMVA"][0])+"\n")
        f.write(';\n')
        f.close()

    def printnetwork(self):
        f = open(self.datfile, 'a')
        f.write('set LE:=\n 1 \n 2;\n')
        #set of transmission lines
        f.write('set L:=\n')
        for i in self.data["branch"].index.tolist():
            f.write(str(self.data["branch"]["name"][i])+"\n")
        f.write(';\n')
        #set of transformers
        if not(self.data["transformer"].empty):
            f.write('set TRANSF:= \n')
            for i in self.data["transformer"].index.tolist():
                f.write(str(self.data["transformer"]["name"][i])+"\n")
            f.write(';\n')
        #---set of generator-bus mapping (gen_bus, gen_ind)---
        f.write('set Gbs:=\n')
        for i in self.data["generator"].index.tolist():
            f.write(str(self.data["generator"]["busname"][i]) + " "+str(self.data["generator"]["name"][i])+"\n")
        f.write(';\n')
        #---set of res generator-bus mapping (resgen_bus, gen_ind)---
        if not(self.data["res"].empty):
            f.write('set Wbs:=\n')
            for i in self.data["res"].index.tolist():
                f.write(str(self.data["res"]["busname"][i]) + " "+str(self.data["res"]["name"][i])+"\n")
            f.write(';\n')
        #---set of demand-bus mapping (demand_bus, demand_ind)---
        f.write('set Dbs:=\n')
        for i in self.data["demand"].index.tolist():
            f.write(str(self.data["demand"]["busname"][i]) + " "+str(self.data["demand"]["name"][i])+"\n")
        f.write(';\n')
        #---set of reference bus---
        f.write('set b0:=\n')
        slackbus = self.data["generator"]["busname"][self.data["generator"]["type"]==3].tolist()
        for i in slackbus:
            f.write(str(i)+""+"\n")
        f.write(';\n')
        #---param defining system topolgy---
        f.write('param A:=\n')
        for i in self.data["branch"].index.tolist():
            f.write(str(self.data["branch"]["name"][i])+" "+"1"+" "+str(self.data["branch"]["from_busname"][i])+"\n")
        for i in self.data["branch"].index.tolist():
            f.write(str(self.data["branch"]["name"][i])+" "+"2"+" "+str(self.data["branch"]["to_busname"][i])+"\n")
        f.write(';\n')
        #---Transformers---
        if not(self.data["transformer"].empty):
            f.write('param AT:= \n')
            for i in self.data["transformer"].index.tolist():
                f.write(str(self.data["transformer"]["name"][i])+" "+"1"+" "+str(self.data["transformer"]["from_busname"][i])+"\n")
            for i in self.data["transformer"].index.tolist():
                f.write(str(self.data["transformer"]["name"][i])+" "+"2"+" "+str(self.data["transformer"]["to_busname"][i])+"\n")
            f.write(';\n')
        f.close()

    def printDC(self):
        f = open(self.datfile, 'a')
        #---Tranmission line chracteristics for DC load flow---
        f.write('param BL:=\n')
        for i in self.data["branch"].index.tolist():
            f.write(str(self.data["branch"]["name"][i])+" "+str(-1/float(self.data["branch"]["x"][i]))+"\n")
        f.write(';\n')
        #---Transformer chracteristics---
        if not(self.data["transformer"].empty):
            f.write('param BLT:=\n')
            for i in self.data["transformer"].index.tolist():
                f.write(str(self.data["transformer"]["name"][i])+" "+str(-float(1/self.data["transformer"]["x"][i]))+"\n")
            f.write(';\n')
        f.close()
    def printOPF(self):
        f = open(self.datfile, 'a')
        #---Real power generation bounds---
        f.write('param PGmin:=\n')
        for i in self.data["generator"].index.tolist():
            f.write(str(self.data["generator"]["name"][i])+" "+str(float(self.data["generator"]["PGLB"][i])/self.data["baseMVA"]["baseMVA"][0])+"\n")
        f.write(';\n')
        f.write('param PGmax:=\n')
        for i in self.data["generator"].index.tolist():
            f.write(str(self.data["generator"]["name"][i])+" "+str(float(self.data["generator"]["PGUB"][i])/self.data["baseMVA"]["baseMVA"][0])+"\n")
        f.write(';\n')
        #---Real power Renewable generation bounds---
        if not(self.data["res"].empty):
            f.write('param WGmin:=\n')
            for i in self.data["res"].index.tolist():
                f.write(str(self.data["res"]["name"][i])+" "+str(float(self.data["res"]["PGLB"][i])/self.data["baseMVA"]["baseMVA"][0])+"\n")
            f.write(';\n')
            f.write('param WGmax:=\n')
            for i in self.data["res"].index.tolist():
                f.write(str(self.data["res"]["name"][i])+" "+str(float(self.data["res"]["PGUB"][i])/self.data["baseMVA"]["baseMVA"][0])+"\n")
            f.write(';\n')
        #---Tranmission line bounds---
        f.write('param SLmax:=\n')
        for i in self.data["branch"].index.tolist():
            f.write(str(self.data["branch"]["name"][i])+" "+str(float(self.data["branch"]["ContinousRating"][i])/self.data["baseMVA"]["baseMVA"][0])+"\n")
        f.write(';\n')
        #---Transformer chracteristics---
        if not(self.data["transformer"].empty):
            f.write('param SLmaxT:=\n')
            for i in self.data["transformer"].index.tolist():
                f.write(str(self.data["transformer"]["name"][i])+" "+str(float(self.data["transformer"]["ContinousRating"][i])/self.data["baseMVA"]["baseMVA"][0])+"\n")
            f.write(';\n')
        #---cost data---
        f.write('param c2:=\n')
        for i in self.data["generator"].index.tolist():
            f.write(str(self.data["generator"]["name"][i])+" "+str(float(self.data["generator"]["costc2"][i]))+"\n")
        f.write(';\n')
        f.write('param c1:=\n')
        for i in self.data["generator"].index.tolist():
            f.write(str(self.data["generator"]["name"][i])+" "+str(float(self.data["generator"]["costc1"][i]))+"\n")
        f.write(';\n')
        f.write('param c0:=\n')
        for i in self.data["generator"].index.tolist():
            f.write(str(self.data["generator"]["name"][i])+" "+str(float(self.data["generator"]["costc0"][i]))+"\n")
        f.write(';\n')
        # --- RES curtailment penalty ---
        if not(self.data["res"].empty):
            f.write('param bid:=\n')
            for i in self.data["res"].index.tolist():
                f.write(str(self.data["res"]["name"][i])+" "+str(float(self.data["res"]["bid"][i]))+"\n")
            f.write(';\n')
        f.close()
    def printDCOPF(self):
        f = open(self.datfile, 'a')
        #---Tranmission line chracteristics---
        f.write('param BL:=\n')
        for i in self.data["branch"].index.tolist():
            f.write(str(self.data["branch"]["name"][i])+" "+str(-1/float(self.data["branch"]["x"][i]))+"\n")
        f.write(';\n')
        #---Transformer chracteristics---
        if not(self.data["transformer"].empty):
            f.write('param BLT:=\n')
            for i in self.data["transformer"].index.tolist():
                f.write(str(self.data["transformer"]["name"][i])+" "+str(-float(1/self.data["transformer"]["x"][i]))+"\n")
            f.write(';\n')
        f.close()
    
    def printSCdat(self):
        flag_C = 0
        contingencies_id = 1
        contingencies_set = []
        ##--Security constrained data--
        f = open(self.datfile, 'a')
        if flag_C==1:
            f.write(';\n')
        ##--Branch contingencies--
        flag_C = 0
        for i in self.data["branch"].index.tolist():
            if float(self.data["branch"]["contingency"][i]) != 0: #2026.1.29 Modified to support any str.

                if flag_C==0:
                    f.write('set CL:=\n')
                    flag_C=1
                f.write(str(contingencies_id)+" "+str(self.data["branch"]["name"][i])+"\n")
                contingencies_set.append([contingencies_id,str(self.data["branch"]["probability"][i])])
                contingencies_id += 1
        if flag_C==1:
            f.write(';\n')
        ##--Transformer contingencies--
        flag_C = 0
        for i in self.data["transformer"].index.tolist():
            if float(self.data["transformer"]["contingency"][i]) != 0: #2026.1.29 Modified to support any str.
                if flag_C==0:
                    f.write('set CT:=\n')
                    flag_C=1
                f.write(str(contingencies_id)+" "+str(self.data["transformer"]["name"][i])+"\n")
                contingencies_set.append([contingencies_id,str(self.data["transformer"]["probability"][i])])
                contingencies_id += 1
        if flag_C==1:
            f.write(';\n')

        ##--set of contingencies--
        probok = 1- sum(float(x[1]) for x in contingencies_set)
        print ('The probability of system being OK is: ',str(probok*100),'%')
        f.write('param probOK:=\n')
        f.write(str(probok) + "\n")
        f.write(';\n')
        f.write('set C:=\n')
        for i in contingencies_set:
            f.write(str(i[0])+"\n")
        f.write(';\n')
        if contingencies_set:
            f.write('param probC:=\n')
            for i in contingencies_set:
                f.write(str(i[0])+" "+str(i[1])+"\n")
            f.write(';\n')
        f.close()
