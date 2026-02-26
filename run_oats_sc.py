#==================================================================
# ---Author---
# W. Bukhsh,
# wbukhsh@gmail.com
# OATS
# Copyright (c) 2015 by W Bukhsh, Glasgow, Scotland
# OATS is distributed under the GNU GENERAL PUBLIC LICENSE v3 (see LICENSE file for details).
#==================================================================
# --- Modifications ---
# Copyright (c) 2026 R. He
# Project: OATSMCS
# Minor interface adaptation for single-scenario preventive DC-SCOPF call.
# Date: Feb 2026
#==================================================================
from oats_sc.run_sc import SCOPF_PREVENTIVE

# IEEE-24R: Derived from the IEEE 24-bus RTS benchmark with updated line/RES data and preventive DC-SCOPF formulation.
tc= 'IEEE-24R/IEEE-24R.xlsx'

SCOPF_PREVENTIVE(solver='gurobi',tc= tc) 


