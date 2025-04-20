import os, sys

# 👇 your DLL + sys.path setup here (as before)
os.add_dll_directory(r"C:\Users\eylan\Documents\antenna_sim\openEMS_v0.0.36\openEMS")
sys.path.insert(0, r"C:\Users\eylan\Documents\antenna_sim\openEMS_v0.0.36\openEMS\python")


from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0
import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'Simp_Patch')
# 1) Configure FDTD (excitation + boundaries)
f0, fc = 2.4e9, 0.5e9
FDTD = openEMS(NrTS=1000, EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PEC']*6)

# 2) Build geometry & mesh
CSX  = ContinuousStructure()
grid = CSX.GetGrid()
grid.SetDeltaUnit(1)       # units in meters
grid.AddLine('x', [-0.06, 0.0, 0.06])
grid.AddLine('y', [-0.005, 0.0, 0.005])
grid.AddLine('z', [-0.005, 0.0, 0.005])

block = CSX.AddMetal('block')
block.AddBox(
    start=[-0.01, -0.001, -0.001],
    stop =[ 0.01,  0.001,  0.001],
    priority=1
)

# 3) Attach CSX to FDTD
FDTD.SetCSX(CSX)

# 4) Add a NF2FF box so Run() has something to bake
nf2ff = FDTD.CreateNF2FFBox()

# 5) Run simulation
out = "sim_debug2"
if os.path.exists(out):
    os.system(f"rmdir /S /Q {out}")
# FDTD.Run(out)
FDTD.Run(Sim_Path, verbose=3, cleanup=True)

print("✅ debug2: completed — check folder 'sim_debug2' for output")