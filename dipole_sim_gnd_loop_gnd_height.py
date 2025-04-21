import os
import sys
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import csv

import configparser
import re

# import config file, config.conf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.conf")
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
OPENEMS_PATH = config.get("openEMS", "path_openems")
OPENEMS_OUTPUT_PATH = config.get("openEMS_output", "path_openEMS_outputs")
OPENEMS_SIM_OUTPUT_PATH = config.get("openEMS_output", "path_openEMS_outputs_sim")
OPENEMS_DATA_OUTPUT_PATH = config.get("openEMS_output", "path_openEMS_outputs_data")
# ─── DLL & Python‑wrapper setup ───────────────────────────────────────────────
os.add_dll_directory(OPENEMS_PATH)
sys.path.insert(0, os.path.join(OPENEMS_PATH,'python'))
# ──────────────────────────────────────────────────────────────────────────────
from pylab import *
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *
from CSXCAD import AppCSXCAD_BIN

# local
import export_vtk as export_vtk
style.use('ggplot')

show_xml = False
include_gnd = True
output_3d = True
plot_3d = True
fast_sim = False
### General parameter setup
sim_dir = OPENEMS_SIM_OUTPUT_PATH
data_dir = os.path.join(OPENEMS_DATA_OUTPUT_PATH, 'dipole_sim')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
master_log_path = os.path.join(data_dir, 'resonance_gnd_height.csv')

# Only create and write header if file doesn't exist
with open(master_log_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Length (mm)', 'Fres (MHz)','Ftarget (MHz)', 'D (dB)','Efficiency (%)',
                        'Gd (dBi)', 'Gr (dBi)', 'GND Distance (mm)', 'GND Division (λ/[gnd_div])'])

print('Created master log file:', master_log_path)

def run_dipole_sim(dipole_length=150, gnd_division=2, Ftarget_MHz=864.5):
    Sim_Path = os.path.join(sim_dir, f'SimData_Dipole_L{dipole_length}mm_GNDdiv{int(gnd_division)}')
    if not os.path.exists(Sim_Path):
        os.makedirs(Sim_Path)

    save_img_path = os.path.join(data_dir, f'{dipole_length}mm_GNDdiv{gnd_division}')
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    
    temp_log = os.path.join(save_img_path, 'resonance_gnd_height.csv')
    # Only create and write header if file doesn't exist
    with open(temp_log, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Length (mm)', 'Fres (MHz)','Ftarget (MHz)', 'D (dB)','Efficiency (%)',
                            'Gd (dBi)', 'Gr (dBi)', 'GND Distance (mm)', 'GND Division (λ/[gnd_div])'])
    # ─── Dipole geometry (all lengths in mm) ──────────────────────────────────────
    # For a 1 GHz dipole: free‑space λ ≃ 300 mm → half‑wave ≃150 mm
    dipole_diameter = 1    # diameter of the “wire” (mm)
    radius   =  dipole_diameter / 2.0
    # ─── Feed settings ────────────────────────────────────────────────────────────
    feed_pos = 0           # feed at center, x = 0 mm
    feed_R   = 50          # feed resistance (Ω)
    f0 = 900e6 # center frequency
    fc = 500e6 # 20 dB corner frequency
    lambda0 = C0 / f0  # in meters
    lambda0_mm = lambda0 * 1e3  # ≈ 300 mm
    half_len =  dipole_length / 2.0

    # 3) Simulation box (mm)
    air_padding = lambda0_mm #  lambda0_mm/2  # mm (half-wavelength buffer)
    print('Air padding:', air_padding)
    #    at least 1.5× dipole_length in x, small y/z
    # SimBox = np.array([dipole_length + 2*air_padding, 
    #                dipole_length + 2*air_padding, 
    #                dipole_length + 2*air_padding])
    air_padding_tip = lambda0_mm / 2
    air_padding_top = lambda0_mm / 2
    air_padding_back  = lambda0_mm / 2  # increased for ground plane

    SimBox = np.array([
        dipole_length + air_padding_tip,  # X
        dipole_length + air_padding_top + air_padding_back,  # Y: ground behind
        dipole_length + air_padding_tip   # Z
    ])
    # 4) FDTD setup
    # setup FDTD parameter & excitation function
    #   - NrTS: maximum timesteps
    #   - EndCriteria: stop when |ΔE/E| drops below 1e‑4 (~‑40 dB)
    #   - Gaussian pulse centered at f0 with bandwidth fc
    if fast_sim:
        # fast simulation with 1/10 of the time steps
        FDTD = openEMS(NrTS=15000, EndCriteria=1e-3)
    else:
        FDTD = openEMS(NrTS=60000, EndCriteria=1e-4) # FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite( f0, fc )
    # Mur absorbing boundaries on all six sides (good for dipole in free space)
    FDTD.SetBoundaryCond(['MUR'] * 6)

    # 5) Geometry & mesh
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX) # attach CAD to solver
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3) # your SimBox is in mm, so 1 unit = 1 mm
    mesh_res_factor = 20  # increase to 30 or 40 for finer resolution
    mesh_res = C0/(f0+fc)/1e-3/mesh_res_factor # fine enough to resolve the pulse

    # Air‑box mesh: at least 3 points per axis  
    # fine region around the dipole
    # coarse far‑field regions
    
    x1 = np.linspace(-half_len-air_padding, -20, 17)   
    x3 = np.linspace(20, half_len+air_padding, 17)   # 17 pts from +20 to +150 mm
    # fine region around the wire
    x2 = np.linspace( -20,  20, 41)   # 41 pts → 1 mm spacing
    # combine, sort & uniquify
    xlin = np.unique(np.hstack((x1, x2, x3)))
    mesh.AddLine('x', xlin)

    # Y-direction mesh
    y_fine = np.linspace(-radius*3, radius*3, 7)  # 7 points → ~0.5 mm spacing
    y_coarse_neg = np.linspace(-dipole_length, -radius*3, 20)
    y_coarse_pos = np.linspace(radius*3, dipole_length, 20)
    y_all = np.unique(np.hstack([y_coarse_neg, y_fine, y_coarse_pos]))
    mesh.AddLine('y', y_all)
    # Z-direction mesh (same as Y)
    mesh.AddLine('z', y_all)
    # 6) Dipole metal
    #   - dipole_length and dipole_diameter are in mm    
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
    if include_gnd:
        gnd_size = air_padding/2  # in mm
        gnd_offset = lambda0_mm/gnd_division # in mm;  lambda0_mm / 2 
        print('*** Ground plane offset:', gnd_offset, ' mm')
        gnd_y = -gnd_offset
        mesh.AddLine('y', [gnd_y])  # align Yee grid
        # Add finite conductivity ground plane
        copper_conductivity = 5.8e7  # S/m
        per_cu = 100.0 / 100.0  # convert to fraction
        gnd_conductivity = copper_conductivity * per_cu  # S/m
        print('*** Ground plane conductivity:', gnd_conductivity, ' S/m')

        Rs = 1 / (gnd_conductivity )
        print(f"*** Surface resistance Rs = {Rs:.3f} Ω/sq")
        gnd = CSX.AddMetal('gnd_plane')
        # gnd = CSX.AddMetal('gnd_plane')
        gnd.AddBox(
            [-gnd_size, gnd_y, -gnd_size/2],   # [X_start, Y, Z_start]
            [ gnd_size, gnd_y,  gnd_size/2]    # [X_stop,  Y, Z_stop]
        )

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
    mesh.SmoothMeshLines('all', mesh_res, 1.3)
    nf2ff_box_padding = lambda0_mm / 2  # or 0.75 * lambda0_mm

    # Define NF2FF dump box BEFORE FDTD.CreateNF2FFBox()
    # nf2ff_dump = CSX.AddDump('nf2ff')
    # nf2ff_dump.SetDumpMode(1)  # 1 = far-field NF2FF box
    # nf2ff_dump.SetDumpType(1)  # 1 = boxed volume

    # nf2ff_dump.AddBox(
    #     [-dipole_length/2 - nf2ff_box_padding,
    #     -lambda0_mm/gnd_division - nf2ff_box_padding,
    #     -dipole_length/2 - nf2ff_box_padding],
        
    #     [dipole_length/2 + nf2ff_box_padding,
    #     +dipole_length/2 + nf2ff_box_padding,
    #     +dipole_length/2 + nf2ff_box_padding]
    # )
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
    # Export E-field vector data in the X–Y plane at Z=0    
    dump_E = CSX.AddDump('Efield_XY')
    dump_E.SetDumpType(0)             # 0 = 2D slice
    dump_E.SetDumpMode(0)             # 0 = vector field
    dump_E.SetNormalDir(2)   # Plane normal axis = Z (0=x, 1=y, 2=z)
    dump_E.AddAttribute('Refinement', str(0.0)) # Convert float to string    

    # Define the spatial region for the dump (XY plane at Z=0)
    dump_size = max(SimBox)  # Make the dump region cover the simulation area in XY
    dump_thickness = max(mesh_res / 2, 0.1)  # force at least 0.1 mm

    dump_E.AddBox(
        [-dump_size/2, -dump_size/2, -dump_thickness],
        [dump_size/2, dump_size/2, dump_thickness]
    )
    print("Dump size in XY:", dump_size, "Dump thickness:", dump_thickness)

    FDTD.Run(Sim_Path, verbose=3, cleanup=True)
    # try:
    #     result = export_vtk.export_E_to_vtk(Sim_Path)
    #     if result == 0:
    #         print("[i] Skipping further VTK processing.")
    # except FileNotFoundError as e:
    #     print(f"[!] Warning: E-field dump missing at {Sim_Path}. Skipping export.")

    ### Post-processing and plotting
    f = np.linspace(max(0.5e9,f0-fc),f0+fc,401)
    port.CalcPort(Sim_Path, f)
    s11 = port.uf_ref/port.uf_inc
    s11_dB = 20.0*np.log10(np.abs(s11))
    idx_min = np.argmin(s11_dB)
    f_res = f[idx_min]
    # s11_dB_val = s11_dB[idx]
    # s11_val = s11[idx]
    # mismatch_loss_factor = 1-np.abs(s11_val)**2

    # get s11 at Ftarget_MHz
    f_target = Ftarget_MHz * 1e6
    idx_target = np.argmin(np.abs(f - f_target))
    s11_dB_target = s11_dB[idx_target]
    s11_target = s11[idx_target]
    figure(figsize=(10, 6))
    plot(f/1e6, s11_dB, 'k-', linewidth=2, label='$S_{11}$')
    legend()
    ylabel('S-Parameter (dB)')
    xlabel('Frequency (MHz)')
    title('Dipole Antenna S11\nFres:{} MHz'.format(f_res/1e6))
    # Label min S11 (resonant freq)
    min_freq = f_res / 1e6
    min_val = s11_dB[idx_min]
    annotate(f'{min_val:.2f} dB\n(Fres)', 
            xy=(min_freq, min_val), 
            xytext=(min_freq + 10, min_val + 3),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10, color='blue')
    # Label S11 at target frequency
    annotate(f'{s11_dB_target:.2f} dB\n(Ftarget)', 
            xy=(Ftarget_MHz, s11_dB_target), 
            xytext=(Ftarget_MHz - 20, s11_dB_target + 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red')

    savefig(os.path.join(save_img_path, 'Dipole_S11.png'))
    tight_layout()  

    Zin = port.uf_tot/port.if_tot    
    Z_res = Zin[idx_min]
    # Z_match = np.conj(Z_res)
    Z_match = np.complex128(75.93554472628661+7.655457478018017j)
    # recompute S11 with Z_match
    
    s11_dB = 20.0 * np.log10(np.abs(s11))
    figure(figsize=(10, 6))
    plot(f/1e6, s11_dB, 'k-', linewidth=2, label='$S_{11}$-Matched')
    legend()
    ylabel('S-Parameter (dB)')
    xlabel('Frequency (MHz)')
    title('Dipole Antenna S11\nFmatch:{} MHz'.format(Ftarget_MHz))
    # Label min S11 (resonant freq)
    min_freq = f_res / 1e6
    min_val = s11_dB[idx_min]
    target_val = s11_dB[idx_target]
    
    annotate(f'{min_val:.2f} dB\n(Fres)', 
            xy=(min_freq, min_val), 
            xytext=(min_freq + 10, min_val + 3),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10, color='blue')
    # Label S11 at target frequency
    annotate(f'{target_val:.2f} dB\n(Ftarget)', 
            xy=(Ftarget_MHz, target_val), 
            xytext=(Ftarget_MHz - 20, target_val + 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red')

    savefig(os.path.join(save_img_path, 'Dipole_S11_Matched.png'))
    tight_layout()  
    print('Target frequency: {} MHz'.format(f_target/1e6))
    print('Length: {} mm'.format(dipole_length))
    print('Resonant frequency: {} MHz'.format(f_res/1e6))
    # get s11 at Ftarget_MHz
    s11_dB_target = s11_dB[idx_target]
    s11_target = s11[idx_target]
    mismatch_loss_factor_target = 1-np.abs(s11_target)**2
      
    print('S11 target: {} dB'.format(s11_dB_target))
    print('S11 target: {}'.format(s11_target)) 
    print('mismatch_loss_factor_target: {}'.format(mismatch_loss_factor_target))

    s11_dB_res = s11_dB[idx_min]
    s11_res = s11[idx_min]
    mismatch_loss_factor_res = 1-np.abs(s11_res)**2

    print('S11 reonant: {} dB'.format(s11_dB_res))
    print('S11 reonant: {}'.format(s11_res)) 

    # output the length and resonant frequency to file

    title_val = f"Zin:{Z_res.real:.2f}+j{Z_res.imag:.2f}Ohm"
    print(title_val)
    figure(figsize=(10, 6))
    plot(f/1e6, np.real(Zin), 'k-', linewidth=2, label='$\Re\{Z_{in}\}$')
    plot(f/1e6, np.imag(Zin), 'r--', linewidth=2, label='$\Im\{Z_{in}\}$')
    legend()
    ylabel('Zin (Ohm)')
    xlabel('Frequency (MHz)')
    title(f'Dipole Antenna Input Impedance\n{title_val}')
    savefig(os.path.join(save_img_path, 'Simp_Patch_Antenna.png'))
    tight_layout()
    
    theta = np.arange(0.0, 181.0, 1.0)   # 0° to 180°
    phi   = np.arange(0.0, 361.0, 1.0)   # 0° to 360°

    # The NF2FF transform applies an integral over a surface enclosing the antenna to compute the far-field:
    # nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f_res, theta, phi, center=[0,0,1e-3])   
    # D_res = nf2ff_res.Dmax[0]  
    # # converts the normalized far-field magnitude into directivity in dBi,
    # E_norm_res = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + D_res
    # efficiency_res = 100 * nf2ff_res.Prad[0] / port.P_acc[idx_min]
    # Gc_res = D_res + 10 * np.log10(efficiency_res / 100)
    # Gr_res = 10 * np.log10(10**(Gc_res / 10) * mismatch_loss_factor_res)
    
    # print(f"Efficiency = {efficiency_res:.2f}%")
    # print(f"Gain_res = {Gc_res:.2f} dBi")
    # print(f"Realized Gain = {Gr_res:.2f} dBi")

    # print(f"Dmax = {nf2ff_res.Dmax[0]:.2f} dBi")
    # print(f"Prad = {nf2ff_res.Prad[0]:.4e} W")
    # print(f"Pacc = {port.P_acc[idx_min]:.4e} W")

    nf2ff_target = nf2ff.CalcNF2FF(Sim_Path, f_target, theta, phi, center=[0,0,1e-3])   
    D_target = nf2ff_target.Dmax[0]  
    # converts the normalized far-field magnitude into directivity in dBi,
    E_norm_target = 20.0*np.log10(nf2ff_target.E_norm[0]/np.max(nf2ff_target.E_norm[0])) + D_target
    # Prad​ : total radiated power (from NF2FF surface integration)
    # Pacc : power accepted by the antenna port
    efficiency_target = 100 * nf2ff_target.Prad[0] / port.P_acc[idx_target]
    Gc_target = D_target + 10 * np.log10(efficiency_target / 100)
    Gr_target = 10 * np.log10(10**(Gc_target / 10) * mismatch_loss_factor_target)
    
    print(f"Efficiency = {efficiency_target:.2f}%")
    print(f"Gain_target = {Gc_target:.2f} dBi")
    print(f"Gr_target = {Gr_target:.2f} dBi")

    print(f"Dmax = {nf2ff_target.Dmax[0]:.2f} dBi")
    print(f"Prad = {nf2ff_target.Prad[0]:.4e} W")
    print(f"Pacc = {port.P_acc[idx_target]:.4e} W")


    with open(master_log_path, mode='a', newline='', encoding='utf-8') as csvfile:
        # Write the data to the CSV file``
        writer = csv.writer(csvfile)
        writer.writerow([f"{dipole_length:.1f}", f"{min_freq:.3f}", f"{Ftarget_MHz:.3f}", f"{D_target:.2f}", f"{efficiency_target:.2f}",
                     f"{Gc_target:.2f}", f"{Gr_target:.2f}", f"{gnd_offset:.2f}", f"{gnd_division:.1f}"])
    
    with open(temp_log, mode='a', newline='', encoding='utf-8') as csvfile:
        # Write the data to the CSV file``
        writer = csv.writer(csvfile)
        writer.writerow([f"{dipole_length:.1f}", f"{min_freq:.3f}", f"{Ftarget_MHz:.3f}", f"{D_target:.2f}", f"{efficiency_target:.2f}",
                     f"{Gc_target:.2f}", f"{Gr_target:.2f}", f"{gnd_offset:.2f}", f"{gnd_division:.1f}"])


    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    phi_index_0 = np.where(phi == 0.0)[0][0]  # Find the correct index for phi = 90°
    phi_index_90 = np.where(phi == 90.0)[0][0]  # Find the correct index for phi = 90°
    added_title_val = f"Fres: {f_res/1e9:.2f} GHz, Eff: {efficiency_target:.2f}%, Gain: {Gc_target:.2f} dBi, Gr: {Gr_target:.2f} dBi"
    # --- Plot phi = 90 (YZ-plane) ---
    figure(figsize=(10, 6))
    ax2 = plt.subplot(1, 1, 1, polar=True)
    ax2.plot(theta_rad, np.squeeze(E_norm_target[:, phi_index_90]), 'r--', linewidth=2, label='phi = 90°')
    ax2.set_title(f'Azimuth Pattern, H-Field (phi = 90°)\n{added_title_val}')
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_rlim(bottom=np.min(E_norm_target)-3, top=np.max(E_norm_target)+3)
    ax2.legend(loc='lower right')
    # Annotate the maximum value for phi = 90
    max_val_phi90 = np.max(np.squeeze(E_norm_target[:, phi_index_90]))
    max_idx_phi90 = np.argmax(np.squeeze(E_norm_target[:, phi_index_90]))
    max_theta_rad_phi90 = theta_rad[max_idx_phi90]
    ax2.annotate(f'{max_val_phi90:.2f} dBi',
                 xy=(max_theta_rad_phi90, max_val_phi90),
                 xytext=(max_theta_rad_phi90 - np.pi/8, max_val_phi90 + 5),  # Adjust text position
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3, headlength=5),
                 fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='bottom')
    tight_layout()
    savefig(os.path.join(save_img_path, 'Azimuth_phi90.png'))

    # θ = 90° → horizontal cut (H-plane)
    theta_idx_90 = np.argmin(np.abs(theta - 90))
    E_theta_90 = np.squeeze(E_norm_target[theta_idx_90, :])  # E(φ) at θ = 90°
    phi_rad = np.radians(phi)

    figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1, polar=True)
    ax.plot(phi_rad, E_theta_90, 'b-', linewidth=2, label='θ = 90°')
    ax.set_title(f'Elevation Pattern, E-field (θ = 90°)\n{added_title_val}')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(bottom=np.min(E_theta_90)-3, top=np.max(E_theta_90)+3)
    ax.legend(loc='lower right')

    # Annotate max
    max_val_phi = np.max(E_theta_90)
    max_idx_phi = np.argmax(E_theta_90)
    max_phi_rad = phi_rad[max_idx_phi]
    ax.annotate(f'{max_val_phi:.2f} dBi',
                xy=(max_phi_rad, max_val_phi),
                xytext=(max_phi_rad + np.pi/8, max_val_phi + 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3, headlength=5),
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='bottom')

    tight_layout()
    plt.savefig(os.path.join(save_img_path, 'Azimuthal_theta90.png'), dpi=300)

    # list all the phi values with max values
    #initialize the csv file
    with open(os.path.join(save_img_path, 'max_values.csv'), 'w') as file_write:
        file_write.write('phi,max_value,max_theta\n')
    for i, phi_val in enumerate(phi):
        E_phi = np.squeeze(E_norm_target[:, i])
        max_val = np.max(E_phi)
        max_idx = np.argmax(E_phi)
        max_theta = theta[max_idx]
        # Find -3 dB points
        half_power_level = max_val - 3.0
        above_half_power = np.where(E_phi >= half_power_level)[0]
        if len(above_half_power) >= 2:
            # HPBW is the angular distance between first and last above -3 dB
            hpbw = theta[above_half_power[-1]] - theta[above_half_power[0]]
        else:
            hpbw = 0.0  # Fallback if beam is too narrow or clipped

        # print(f"phi = {phi_val:.1f}°: max = {max_val:.2f} dBi at theta = {max_theta:.1f}°, HPBW = {hpbw:.1f}°")
        with open(os.path.join(save_img_path, 'max_values.csv'), 'a') as file_write:
            file_write.write(f"{phi_val:.1f},{max_val:.2f},{max_theta:.1f},{hpbw:.1f}\n")
    # Save the full far-field data for postprocessing
    np.savez(os.path.join(save_img_path, 'far_field_data.npz'),
            E_norm=E_norm_target,
            theta=theta,
            phi=phi,
            f_res=f_res)

    
    return Sim_Path, save_img_path


if __name__ == "__main__":
    # gnd_divisions = [1, 2,4,8,16,32, 64, 128 ]
    gnd_divisions = [2,4,8,16,32, 64, 128 ]
    # gnd_divisions = [8]
    for gnd_division in gnd_divisions:
        try:
            Sim_Path, save_img_path = run_dipole_sim(dipole_length=150, gnd_division=gnd_division)
        except Exception as e:
            print(f"[!] Error during sim/export at GNDdiv={gnd_division}: {e}")
            continue
    if output_3d:
        export_vtk.write_pvd_wrapper(Sim_Path)
    if plot_3d:
        # save_img_path = os.path.join(data_dir, '150mm')
        export_vtk.far_field_data_npz(np.load(os.path.join(save_img_path, 'far_field_data.npz')), save_img_path)
