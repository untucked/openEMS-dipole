import matplotlib.pyplot as plt
import numpy as np
import os

def plot_Zin(f, Zin, title_val, save_img_path):
    plt.figure(figsize=(10, 6))
    plt.plot(f/1e6, np.real(Zin), 'k-', linewidth=2, label='$\Re\{Z_{in}\}$')
    plt.plot(f/1e6, np.imag(Zin), 'r--', linewidth=2, label='$\Im\{Z_{in}\}$')
    plt.legend()
    plt.ylabel('Zin (Ohm)')
    plt.xlabel('Frequency (MHz)')
    plt.title(f'Dipole Antenna Input Impedance\n{title_val}')
    plt.savefig(os.path.join(save_img_path, 'Simp_Patch_Antenna.png'))
    plt.tight_layout()

def plot_S11(f, s11_dB, title_val, save_img_path,
             f_res, idx_res,
             Ftarget_MHz, target_val):
    plt.figure(figsize=(10, 6))
    plt.plot(f/1e6, s11_dB, 'k-', linewidth=2, label='$S_{11}$-Matched')
    plt.legend()
    plt.ylabel('S-Parameter (dB)')
    plt.xlabel('Frequency (MHz)')
    plt.title(f'Dipole Antenna S11\n{title_val}')

    min_freq = f_res / 1e6
    min_val = s11_dB[idx_res]
    
    plt.annotate(f'{min_val:.2f} dB\n(Fres)', 
            xy=(min_freq, min_val), 
            xytext=(min_freq + 10, min_val + 3),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10, color='blue')
    # Label S11 at target frequency
    plt.annotate(f'{target_val:.2f} dB\n(Ftarget)', 
            xy=(Ftarget_MHz, target_val), 
            xytext=(Ftarget_MHz - 20, target_val + 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red')

    plt.savefig(os.path.join(save_img_path, 'Dipole_S11_Matched.png'))
    plt.tight_layout()  