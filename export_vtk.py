import h5py
import meshio
import os
import numpy as np

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
