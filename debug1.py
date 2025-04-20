import os, sys

# 👇 your DLL + sys.path setup here (as before)
os.add_dll_directory(r"C:\Users\eylan\Documents\antenna_sim\openEMS_v0.0.36\openEMS")
sys.path.insert(0, r"C:\Users\eylan\Documents\antenna_sim\openEMS_v0.0.36\openEMS\python")

from CSXCAD import ContinuousStructure
from openEMS import openEMS

# 1) build the simplest box mesh
CSX  = ContinuousStructure()
grid = CSX.GetGrid()
grid.SetDeltaUnit(1)      # units = meters
grid.AddLine('x', [-0.06, 0.06])
grid.AddLine('y', [-0.005, 0.005])
grid.AddLine('z', [-0.005, 0.005])

# 2) add a simple PEC block (no ports or anything)
metal = CSX.AddMetal('block')
metal.AddBox(
    priority=1,
    start=[-0.01, -0.001, -0.001],
    stop =[ 0.01,  0.001,  0.001],
)

# 3) run with no excitation (just to test geometry)
FDTD = openEMS()
FDTD.SetCSX(CSX)
out = "sim_debug1"
if os.path.exists(out):  os.system(f"rmdir /S /Q {out}")
FDTD.Run(out)

print("debug1: completed (check folder 'sim_debug1')")
