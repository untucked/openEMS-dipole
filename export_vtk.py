import h5py
import meshio
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def export_E_to_vtk(h5_path, vtk_path_prefix='Efield_XY'):
    h5_file = os.path.join(h5_path, f'{vtk_path_prefix}.h5')
    if not os.path.isfile(h5_file):
        print(f"[!] HDF5 dump not found: {h5_file}")
        return

    with h5py.File(h5_file, 'r') as f:
        coords = np.array(f['Grid'])
        Ex = np.array(f['Ex'])
        Ey = np.array(f['Ey'])
        Ez = np.array(f['Ez'])

        points = coords.T
        npts = points.shape[0]

        # Create structured grid indices assuming ordered regular grid
        cells = [("vertex", np.arange(npts).reshape(-1, 1))]

        mesh = meshio.Mesh(
            points=points,
            cells=cells,
            point_data={
                "Ex": Ex,
                "Ey": Ey,
                "Ez": Ez
            }
        )
        vtk_file = os.path.join(h5_path, f"{vtk_path_prefix}.vtk")
        meshio.write(vtk_file, mesh)
        print(f"[✔] Exported: {vtk_file}")

def write_pvd_wrapper(vtr_folder, output_filename="Efield_XY.pvd", timestep_fs=60):
    """
    Generates a ParaView PVD file for loading .vtr time-series data.

    Parameters:
    - vtr_folder: folder containing .vtr files
    - output_filename: the name of the generated .pvd file
    - timestep_fs: time step in femtoseconds (default is ~60 fs for openEMS 15 GHz BW)
    """
    vtr_files = sorted([f for f in os.listdir(vtr_folder) if f.endswith('.vtr')])
    output_path = os.path.join(vtr_folder, output_filename)
    
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        
        for i, file in enumerate(vtr_files):
            time_fs = i * timestep_fs
            f.write(f'    <DataSet timestep="{time_fs*1e-15:.6e}" group="" part="0" file="{file}"/>\n')
        
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    
    print(f"[✔] Wrote PVD file: {output_path}")


def far_field_data_npz(data, save_img_path):
    # Convert to arrays
    theta_deg = np.array(data['theta'])  # e.g. -180 to 180
    phi_deg = np.array(data['phi'])
    # Define cut limits
    theta_min, theta_max = 0, 180
    phi_min, phi_max = 0, 90
    # Get index ranges
    theta_idx = np.where((theta_deg >= theta_min) & (theta_deg <= theta_max))[0]
    phi_idx = np.where((phi_deg >= phi_min) & (phi_deg <= phi_max))[0]

    E_dBi = data['E_norm'][np.ix_(theta_idx, phi_idx)]
    theta_deg = theta_deg[theta_idx]
    phi_deg = phi_deg[phi_idx]
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    f_res = data['f_res']

    # Meshgrid for 3D spherical coordinate transformation
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    R = E_dBi
    D_dBi = E_dBi

    # Subsample for faster plotting
    step_theta = 10  # plot every 2nd point in theta
    step_phi = 10    # plot every 2nd point in phi

    THETA_sub = THETA[::step_theta, ::step_phi]
    PHI_sub   = PHI[::step_theta, ::step_phi]
    R_sub     = R[::step_theta, ::step_phi]
    D_sub     = D_dBi[::step_theta, ::step_phi]
    # Clip dBi for color (floor = max - 20 dB)
    max_gain = np.max(D_sub)
    min_threshold = max_gain - 20
    D_clipped = np.clip(D_sub, min_threshold, max_gain)
    # Normalize color
    norm = plt.Normalize(vmin=min_threshold, vmax=max_gain)
    colors = plt.cm.jet(norm(D_clipped))

    # Spherical → Cartesian
    X = R_sub * np.sin(THETA_sub) * np.cos(PHI_sub)
    Y = R_sub * np.sin(THETA_sub) * np.sin(PHI_sub)
    Z = R_sub * np.cos(THETA_sub)

    # Color normalization
    norm = plt.Normalize(vmin=min_threshold, vmax=max_gain)
    colors = plt.cm.jet(norm(D_clipped))
     # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=colors,
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=False, alpha=0.95)
    ax.set_title(f'3D Far-Field Pattern @ {f_res/1e9:.2f} GHz')
    ax.set_box_aspect([1,1,1])  # equal scaling

    # Colorbar
    mappable = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label='Directivity (dBi)')

    # Save & show
    plt.savefig(os.path.join(save_img_path, "dipole_3D_gain.png"), dpi=300)
    plt.show()