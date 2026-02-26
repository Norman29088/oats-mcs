#==================================================================
# SCOPF_PREVENTIVE.py
# Pyomo implementation of preventive DC Security-Constrained Optimal Power Flow (SCOPF)
#
# --- Original Author ---
# W. Bukhsh (OATS)
# Copyright (c) 2017 W. Bukhsh
# Licensed under the GNU General Public License v3.0 (GPLv3).
#
# --- Modifications ---
# Copyright (c) 2026 R. He
# Project: OATS-MCS (Monte Carlo Sampling extension for OATS)
# Adapted for preventive DC-SCOPF formulation.
# Date: Feb 2026
#==================================================================


#==========Import==========
from __future__ import division
from pyomo.environ import *
#==========================

model = AbstractModel()

# --- SETS ---
model.B      = Set()  # set of buses, as a list of positive integers
model.G      = Set()  # set of generators, as a list of positive integers
model.RES   = Set()  # set of res generators, as a list of positive integers
model.D      = Set()  # set of demands, as a list of positive integers
model.L      = Set()  # set of lines, as a list of positive integers
model.TRANSF = Set()  # set of transformers, as a list of positive integers
model.LE     = Set()  # line-to and from ends set (1,2)
model.b0     = Set(within=model.B)  # set of reference buses
model.C      = Set()  # set of contigencies

# set of contingencies
model.CL     = Set(within=model.C * model.L)# set of line contigencies
model.CT     = Set(within=model.C * model.TRANSF)# set of transformer contigencies

# generators, buses, loads linked to each bus b
model.Gbs     = Set(within=model.B * model.G)    # set of generator-node mapping
model.Wbs     = Set(within=model.B * model.RES) # set of res-node mapping
model.Dbs     = Set(within=model.B * model.D)    # set of demand-node mapping

# --- parameters ---
# line matrix
model.A     = Param(model.L*model.LE,within=Any)       # bus-line (node-arc) matrix
model.AT    = Param(model.TRANSF*model.LE,within=Any)  # bus-transformer (node-arc) matrix

# demands
model.PD      = Param(model.D, within=Reals)  # real power demand at load d, p.u.
# generators
model.PGmax    = Param(model.G, within=NonNegativeReals)# max real power of generator, p.u.
model.PGmin    = Param(model.G, within=Reals)# min real power of generator, p.u.
model.WGmax    = Param(model.RES, within=NonNegativeReals)# max real power of res generator, p.u.
model.WGmin    = Param(model.RES, within=NonNegativeReals)# min real power of res generator, p.u.

# lines and transformer chracteristics and ratings
model.SLmax  = Param(model.L, within=NonNegativeReals) # max real power limit on flow in a line, p.u.
model.SLmaxT = Param(model.TRANSF, within=NonNegativeReals) # max real power limit on flow in line l, p.u.
model.BL     = Param(model.L, within=Reals)  # susceptance of a line, p.u.
model.BLT    = Param(model.TRANSF, within=Reals)  # susceptance of line l, p.u.

# cost data
model.c2    = Param(model.G, within=NonNegativeReals)# generator cost coefficient c2 (*pG^2)
model.c1    = Param(model.G, within=NonNegativeReals)# generator cost coefficient c1 (*pG)
model.c0    = Param(model.G, within=NonNegativeReals)# generator cost coefficient c0
# model.bid   = Param(model.RES, within=NonNegativeReals, default=0.0)  # curtailment penalty
model.bid = Param(model.RES, within=Reals, default=0.0) #Allow negative penalty

model.baseMVA = Param(within=NonNegativeReals)# base MVA

#constants
model.probC  = Param(model.C, domain=NonNegativeReals) #probabaility of contingencies
model.probOK = Param(domain=NonNegativeReals) #probabaility of the system being OK


# === Pre-contigency variables ===
# --- control variables ---
model.pG    = Var(model.G, domain= Reals)  #real power generation
model.pW    = Var(model.RES, domain= Reals) #real power generation from res
# --- state variables ---
model.deltaL = Var(model.L, domain= Reals) # angle difference across lines
model.deltaLT = Var(model.TRANSF, domain= Reals) # angle difference across transformers
model.delta = Var(model.B, domain= Reals, initialize=0.0) # voltage phase angle at bus b, rad
model.pL = Var(model.L, domain= Reals) # real power injected at b onto line l, p.u.
model.pLT = Var(model.TRANSF, domain= Reals) # real power injected at b onto transformer line l, p.u.


# --- state variables ---
model.pLC      = Var(model.L,model.C, domain= Reals) # real power injected at b onto line l, p.u.
model.pLTC     = Var(model.TRANSF,model.C, domain= Reals) # real power injected at b onto transformer l, p.u.
model.deltaLC  = Var(model.L,model.C, domain= Reals) # angle difference across lines
model.deltaLTC = Var(model.TRANSF,model.C, domain= Reals) # angle difference across transformers
model.deltaC   = Var(model.B,model.C, domain= Reals, initialize=0.0) # voltage phase angle at bus b, rad

# --- pre- and post- contingency costs
model.FPreCont  = Var() # Objective function component for pre-contingency operation

# --- cost function --- 
def objective(model): # New for SCOPF Preventive-2026.1.14
    obj = model.FPreCont
    return obj
model.OBJ = Objective(rule=objective, sense=minimize)

# --- cost components of the objective function ---
def precontingency_cost(model):
    gen_cost = sum(model.c2[g]*(model.baseMVA*model.pG[g])**2+ model.c1[g]*model.baseMVA*model.pG[g]+ model.c0[g]for g in model.G)
    curt_cost = sum(model.bid[w] * (model.WGmax[w] - model.pW[w]) * model.baseMVA for w in model.RES)
    return model.FPreCont == gen_cost + curt_cost
model.precontingency_cost_const = Constraint(rule=precontingency_cost)


# --- Kirchoff's current law Definition at each bus b ---
def KCL_def(model, b):
    return sum(model.pG[g] for g in model.G if (b,g) in model.Gbs) +\
    sum(model.pW[w] for w in model.RES if (b,w) in model.Wbs) ==\
    sum(model.PD[d] for d in model.D if (b,d) in model.Dbs)+\
    sum(model.pL[l] for l in model.L if model.A[l,1]==b)-\
    sum(model.pL[l] for l in model.L if model.A[l,2]==b)+\
    sum(model.pLT[l] for l in model.TRANSF if model.AT[l,1]==b)-\
    sum(model.pLT[l] for l in model.TRANSF if model.AT[l,2]==b)
# the next line creates one KCL constraint for each bus
model.KCL_const = Constraint(model.B, rule=KCL_def)

# --- Kirchoff's voltage law on each line and transformer---
def KVL_line_def(model,l):
    return model.pL[l] == (-model.BL[l])*model.deltaL[l]
def KVL_trans_def(model,l):
    return model.pLT[l] == (-model.BLT[l])*model.deltaLT[l]

#the next two lines creates KVL constraints for each line and transformer, respectively.
model.KVL_line_const     = Constraint(model.L, rule=KVL_line_def)
model.KVL_trans_const    = Constraint(model.TRANSF, rule=KVL_trans_def)


# --- generator power limits ---
def Real_Power_Max(model,g):
    return model.pG[g] <= model.PGmax[g]
def Real_Power_Min(model,g):
    return model.pG[g] >= model.PGmin[g]
#the next two lines creates generation bounds for each generator.
model.PGmaxC = Constraint(model.G, rule=Real_Power_Max)
model.PGminC = Constraint(model.G, rule=Real_Power_Min)

# ---res generator power limits ---
def RES_Real_Power_Max(model,w):
    return model.pW[w] <= model.WGmax[w]
def RES_Real_Power_Min(model,w):
    return model.pW[w] >= model.WGmin[w]
#the next two lines creates generation bounds for each generator.
model.WGmaxC = Constraint(model.RES, rule=RES_Real_Power_Max)
model.WGminC = Constraint(model.RES, rule=RES_Real_Power_Min)

# --- line power limits ---
def line_lim1_def(model,l):
    return model.pL[l] <= model.SLmax[l]
def line_lim2_def(model,l):
    return model.pL[l] >= -model.SLmax[l]
#the next two lines creates line flow constraints for each line.
model.line_lim1 = Constraint(model.L, rule=line_lim1_def)
model.line_lim2 = Constraint(model.L, rule=line_lim2_def)

# --- power flow limits on transformer lines---
def transf_lim1_def(model,l):
    return model.pLT[l] <= model.SLmaxT[l]
def transf_lim2_def(model,l):
    return model.pLT[l] >= -model.SLmaxT[l]
#the next two lines creates line flow constraints for each transformer.
model.transf_lim1 = Constraint(model.TRANSF, rule=transf_lim1_def)
model.transf_lim2 = Constraint(model.TRANSF, rule=transf_lim2_def)

# --- phase angle constraints ---
def phase_angle_diff1(model,l):
    return model.deltaL[l] == model.delta[model.A[l,1]] - \
    model.delta[model.A[l,2]]
model.phase_diff1 = Constraint(model.L, rule=phase_angle_diff1)

# --- phase angle constraints ---
def phase_angle_diff2(model,l):
    return model.deltaLT[l] == model.delta[model.AT[l,1]] - \
    model.delta[model.AT[l,2]]
model.phase_diff2 = Constraint(model.TRANSF, rule=phase_angle_diff2)

# --- reference bus constraint ---
def ref_bus_def(model,b):
    return model.delta[b]==0
model.refbus = Constraint(model.b0, rule=ref_bus_def)

# the next line creates one KCL constraint for each bus under Post-contingency constraints
def KCL_def_PostCnt(model, b, c):
    return (
        sum(model.pG[g] for g in model.G if (b,g) in model.Gbs)
      + sum(model.pW[w] for w in model.RES if (b,w) in model.Wbs)
      == sum(model.PD[d] for d in model.D if (b,d) in model.Dbs)
      + sum(model.pLC[l,c] for l in model.L if model.A[l,1]==b and (c,l) not in model.CL)
      - sum(model.pLC[l,c] for l in model.L if model.A[l,2]==b and (c,l) not in model.CL)
      + sum(model.pLTC[t,c] for t in model.TRANSF if model.AT[t,1]==b and (c,t) not in model.CT)
      - sum(model.pLTC[t,c] for t in model.TRANSF if model.AT[t,2]==b and (c,t) not in model.CT)
    )
model.KCL_const_PostCnt = Constraint(model.B,model.C, rule=KCL_def_PostCnt)

# --- Kirchoff's voltage law on each line ---
def KVL_line_def_PostCnt(model,l,c):
    if (c,l) in model.CL:
        return model.pLC[l,c] == 0
    else:
        return model.pLC[l,c] == (-model.BL[l])*model.deltaLC[l,c]
def KVL_trans_def_PostCnt(model,l,c):
    if (c,l) in model.CT:
        return model.pLTC[l,c] == 0
    else:
        return model.pLTC[l,c] == (-model.BLT[l])*model.deltaLTC[l,c]
#the next two lines create KVL constraints for each line and transformer, respectively.
model.KVL_line_const_PostCnt     = Constraint(model.L,model.C, rule=KVL_line_def_PostCnt)
model.KVL_trans_const_PostCnt    = Constraint(model.TRANSF,model.C, rule=KVL_trans_def_PostCnt)



def line_lim1_def_PostCnt(model, l, c):
    if (c,l) in model.CL:
        return Constraint.Skip
    return model.pLC[l,c] <= model.SLmax[l]

def line_lim2_def_PostCnt(model, l, c):
    if (c,l) in model.CL:
        return Constraint.Skip
    return model.pLC[l,c] >= -model.SLmax[l]
#the next two lines create line flow constraints for each line.
model.line_lim1_PostCnt = Constraint(model.L,model.C, rule=line_lim1_def_PostCnt)
model.line_lim2_PostCnt = Constraint(model.L,model.C, rule=line_lim2_def_PostCnt)

def transf_lim1_def_PostCnt(model, t, c):
    if (c,t) in model.CT:
        return Constraint.Skip
    return model.pLTC[t,c] <= model.SLmaxT[t]

def transf_lim2_def_PostCnt(model, t, c):
    if (c,t) in model.CT:
        return Constraint.Skip
    return model.pLTC[t,c] >= -model.SLmaxT[t]
#the next two lines create line flow constraints for each transformer.
model.transf_lim1_PostCnt = Constraint(model.TRANSF,model.C, rule=transf_lim1_def_PostCnt)
model.transf_lim2_PostCnt = Constraint(model.TRANSF,model.C, rule=transf_lim2_def_PostCnt)

# --- phase angle constraints ---
def phase_angle_diff1_PostCnt(model,l,c):
    return model.deltaLC[l,c] == model.deltaC[model.A[l,1],c] - \
    model.deltaC[model.A[l,2],c]
#the next line creates a constraint to link angle difference across a line.
model.phase_diff1_PostCnt = Constraint(model.L,model.C, rule=phase_angle_diff1_PostCnt)

# --- phase angle constraints ---
def phase_angle_diff2_PostCnt(model,l,c):
    return model.deltaLTC[l,c] == model.deltaC[model.AT[l,1],c] - \
    model.deltaC[model.AT[l,2],c]
#the next line creates a constraint to link angle difference across a transformer.
model.phase_diff2_PostCnt = Constraint(model.TRANSF,model.C, rule=phase_angle_diff2_PostCnt)

# --- reference bus constraint ---
def ref_bus_def_PostCnt(model,b,c):
    return model.deltaC[b,c]==0
model.refbus_PostCnt = Constraint(model.b0,model.C, rule=ref_bus_def_PostCnt)

