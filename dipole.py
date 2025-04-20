import os, sys
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# ─── DLL & Python‑wrapper setup ───────────────────────────────────────────────
os.add_dll_directory(r"C:\Users\eylan\Documents\antenna_sim\openEMS_v0.0.36\openEMS")
sys.path.insert(0, r"C:\Users\eylan\Documents\antenna_sim\openEMS_v0.0.36\openEMS\python")
# ──────────────────────────────────────────────────────────────────────────────

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

# 1) Simulation folder
temp_dir = tempfile.gettempdir()
sim_path = os.path.join(temp_dir, "Dipole_Sim")
if os.path.exists(sim_path):
    os.system(f"rmdir /S /Q {sim_path}")

# 2) FDTD settings
f0, fc = 2.4e9, 0.5e9  # center and bandwidth
FDTD = openEMS(NrTS=2000, EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PEC']*6)

# 3) Geometry & mesh
CSX = ContinuousStructure()
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)  # units in meters

# Mesh: at least 3 points per axis
mesh.AddLine('x', np.linspace(-0.05, 0.05, 21))   # 5 cm half-span
mesh.AddLine('y', np.linspace(-0.001, 0.001, 5))  # 1 mm thickness
mesh.AddLine('z', np.linspace(-0.001, 0.001, 5))

# 4) Add dipole metal (5 cm × 1 mm × 1 mm)
dipole = CSX.AddMetal('dipole')
dipole.AddBox(
    priority=10,
    start=[-0.025, -0.0005, -0.0005],
    stop =[ 0.025,  0.0005,  0.0005],
)

# 5) Attach CAD to solver
FDTD.SetCSX(CSX)

# 6) Snap metal edges & add port
mesh_res = C0/(f0+fc)/1e-3/20
FDTD.AddEdges2Grid(
    dirs='xyz', properties=dipole,
    metal_edge_res=mesh_res/2
)
port = FDTD.AddLumpedPort(
    port_nr=1, R=50,
    start=[0,0,-0.0005], stop=[0,0,0.0005],
    p_dir='z', excite=1
)

# 7) Optional far-field box for radiation pattern
nf2ff = FDTD.CreateNF2FFBox()

# 8) Run the simulation
FDTD.Run(sim_path, verbose=2, cleanup=True)

# 9) Post‑process: compute and plot S11
f = np.linspace(max(1e9, f0-fc), f0+fc, 401)
port.CalcPort(sim_path, f)
s11 = port.uf_ref / port.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))

plt.figure()
plt.plot(f/1e9, s11_dB, '-o')
plt.grid(True)
plt.title('Dipole Return Loss S11')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.ylim(-40, 0)
plt.tight_layout()
plt.show()

# 10) Post‑process: radiation pattern at resonance
# find resonance index
idx = np.argmin(np.abs(s11_dB - np.min(s11_dB)))
f_res = f[idx]
theta = np.arange(-180, 182, 2)
phi   = [0., 90.]
nf2ff_res = nf2ff.CalcNF2FF(sim_path, f_res, theta, phi, center=[0,0,1e-3])

plt.figure()
E_norm = 20 * np.log10(nf2ff_res.E_norm[0] / np.max(nf2ff_res.E_norm[0])) + nf2ff_res.Dmax[0]
plt.plot(theta, np.squeeze(E_norm[:,0]), 'k-', label='xz-plane')
plt.plot(theta, np.squeeze(E_norm[:,1]), 'r--', label='yz-plane')
plt.grid(True)
plt.title(f'Radiation Pattern at {f_res/1e9:.2f} GHz')
plt.xlabel('Theta (deg)')
plt.ylabel('Directivity (dBi)')
plt.legend()
plt.tight_layout()
plt.show()
