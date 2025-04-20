import os, sys
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# ─── DLL & Python‑wrapper setup ───────────────────────────────────────────────
os.add_dll_directory(r"C:\Users\eylan\Documents\antenna_sim\openEMS_v0.0.36\openEMS")
sys.path.insert(0, r"C:\Users\eylan\Documents\antenna_sim\openEMS_v0.0.36\openEMS\python")
# ──────────────────────────────────────────────────────────────────────────────
from pylab import *
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *
from CSXCAD import AppCSXCAD_BIN

style.use('ggplot')

show_xml = False
### General parameter setup
script_directory = os.path.dirname(os.path.abspath(__file__))
Sim_Path = os.path.join(tempfile.gettempdir(), 'Dipole_Patch')

save_img_path = os.path.join(script_directory,'dipole_sim')
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
post_proc_only = False

# ─── Dipole geometry (all lengths in mm) ──────────────────────────────────────
# For a 1 GHz dipole: free‑space λ ≃ 300 mm → half‑wave ≃150 mm
dipole_length = 150    # total dipole length (mm)
dipole_diameter = 1    # diameter of the “wire” (mm)
radius   =  dipole_diameter / 2.0
# ─── Feed settings ────────────────────────────────────────────────────────────
feed_pos = 0           # feed at center, x = 0 mm
feed_R   = 50          # feed resistance (Ω)

# 3) Simulation box (mm)
#    at least 1.5× dipole_length in x, small y/z
SimBox = np.array([300, 200, 200])    # [x, y, z] in mm

# 4) FDTD setup
# setup FDTD parameter & excitation function
f0 = 1e9 # center frequency
fc = 0.5e9 # 20 dB corner frequency
#   - NrTS: maximum timesteps
#   - EndCriteria: stop when |ΔE/E| drops below 1e‑4 (~‑40 dB)
#   - Gaussian pulse centered at f0 with bandwidth fc
FDTD = openEMS(NrTS=15000, EndCriteria=1e-3) # FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
FDTD.SetGaussExcite( f0, fc )
# Mur absorbing boundaries on all six sides (good for dipole in free space)
FDTD.SetBoundaryCond(['MUR'] * 6)

# 5) Geometry & mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX) # attach CAD to solver
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3) # your SimBox is in mm, so 1 unit = 1 mm
mesh_res = C0/(f0+fc)/1e-3/20 # fine enough to resolve the pulse

# Air‑box mesh: at least 3 points per axis  
#   - x: 300 mm span → 51 pts (6 mm spacing)
#   - y: 20 mm span  → 11 pts (2 mm spacing)
#   - z: 20 mm span  → 11 pts (2 mm spacing)
# fine region around the dipole
# coarse far‑field regions
x1 = np.linspace(-150, -20, 17)   # 17 pts from –150 to –20 mm
x3 = np.linspace(  20, 150, 17)   # 17 pts from +20 to +150 mm
# fine region around the wire
x2 = np.linspace( -20,  20, 41)   # 41 pts → 1 mm spacing
# combine, sort & uniquify
xlin = np.unique(np.hstack((x1, x2, x3)))
mesh.AddLine('x', xlin)

# Y-direction mesh
y_fine = np.linspace(-radius*3, radius*3, 7)  # 7 points across 3mm
y_coarse = np.linspace(-100, -radius*3, 10)
mesh.AddLine('y', np.unique(np.hstack([y_coarse, y_fine, -y_coarse])))

# Z-direction mesh (same as Y)
z_fine = np.linspace(-radius*3, radius*3, 7)
z_coarse = np.linspace(-100, -radius*3, 10)
mesh.AddLine('z', np.unique(np.hstack([z_coarse, z_fine, -z_coarse])))


# 6) Dipole metal
#   - dipole_length and dipole_diameter are in mm
half_len =  dipole_length / 2.0
radius   =  dipole_diameter / 2.0
start = [-half_len, 0.0, 0.0]
stop  = [ half_len, 0.0, 0.0]
# a PEC dipole center‑fed at x=0, extending ±half_len in x, radius in y/z
dipole = CSX.AddMetal('dipole')
gap = 1.0  # mm
half_len = dipole_length / 2.0
half_arm = (dipole_length - gap) / 2.0

# Left arm
dipole.AddCylinder(
    [-half_len, 0, 0],
    [-gap/2,    0, 0],
    radius
)

# Right arm
dipole.AddCylinder(
    [gap/2, 0, 0],
    [half_len, 0, 0],
    radius
)

# dipole.SetR(radius)          # set the cylinder radius (in mm because Δ=1e‑3 m)
# dipole.SetMaterial('PEC')    # make it a perfect conductor
# dipole.SetPriority(10)       # same layering priority you had before
FDTD.AddEdges2Grid(
    dirs='xyz',
    properties=dipole,
    metal_edge_res=mesh_res/2
)
# 7) Lumped port feed
# Create 1mm gap in dipole center
gap = 1.0  # 1mm gap
# Force mesh lines so port aligns with grid cells
mesh.AddLine('x', [-gap/2, gap/2])
mesh.AddLine('y', [-radius, radius])
mesh.AddLine('z', [-radius, radius])
start = [-gap/2, 0, 0]
stop  = [ gap/2, 0, 0]
# Create port across the gap
port = FDTD.AddLumpedPort(
    port_nr=1, R=feed_R,
    start=start,
    stop=stop,
    p_dir='x',  # Current flows along X-axis
    excite=1.0,
    priority=5,
    edges2grid='x' # Specify the direction for edge integration

)

# 8) Optional: NF2FF box
mesh.SmoothMeshLines('all', mesh_res, 1.4)
# Add the nf2ff recording box
nf2ff = FDTD.CreateNF2FFBox()

# Add a visualization-only dielectric at the port location (1mm wide box)
debug_vis = CSX.AddMaterial('port_debug')
debug_vis.SetMaterialProperty(epsilon=1.0) 
debug_vis.AddBox([-gap/2, -radius, -radius], [gap/2, radius, radius])
debug_vis.SetColor('#FF0000', 230)  # Red with 90% opacity (255*0.9≈230)


### Run the simulation
CSX_file = os.path.join(save_img_path, 'simp_patch.xml')
CSX.Write2XML(CSX_file)
if show_xml:
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

if not post_proc_only:
    # FDTD.AddDump('E', save_img_path + '/E', dump_type=0)  # type 0 = regular 3D dump
    # FDTD.AddDump('H', save_img_path + '/H', dump_type=0)
    FDTD.Run(Sim_Path, verbose=3, cleanup=True)


### Post-processing and plotting
f = np.linspace(max(0.5e9,f0-fc),f0+fc,401)
port.CalcPort(Sim_Path, f)
s11 = port.uf_ref/port.uf_inc
s11_dB = 20.0*np.log10(np.abs(s11))
out_png = os.path.join(save_img_path, 'Dipole_S11.png')

figure()
plot(f/1e9, s11_dB, 'k-', linewidth=2, label='$S_{11}$')
legend()
ylabel('S-Parameter (dB)')
xlabel('Frequency (GHz)')
savefig(os.path.join(save_img_path, 'Dipole_S11.png'))
tight_layout()  
idx = np.argmin(s11_dB)
f_res = f[idx]
theta = np.arange(-180.0, 180.0, 2.0)
phi   = [0., 90.]
nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f_res, theta, phi, center=[0,0,1e-3])

figure()
E_norm = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + nf2ff_res.Dmax[0]
plot(theta, np.squeeze(E_norm[:,0]), 'k-', linewidth=2, label='xz-plane')
plot(theta, np.squeeze(E_norm[:,1]), 'r--', linewidth=2, label='yz-plane')
ylabel('Directivity (dBi)')
xlabel('Theta (deg)')
title('Frequency: {} GHz'.format(f_res/1e9))
legend()
savefig(os.path.join(save_img_path, 'Simp_Patch_Antenna_Directivity.png'))
tight_layout()

Zin = port.uf_tot/port.if_tot
figure()
plot(f/1e9, np.real(Zin), 'k-', linewidth=2, label='$\Re\{Z_{in}\}$')
plot(f/1e9, np.imag(Zin), 'r--', linewidth=2, label='$\Im\{Z_{in}\}$')
legend()
ylabel('Zin (Ohm)')
xlabel('Frequency (GHz)')
savefig(os.path.join(save_img_path, 'Simp_Patch_Antenna.png'))
tight_layout()
show()
