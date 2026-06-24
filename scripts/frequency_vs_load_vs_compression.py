"""
sweep_freq_2d.py

Prima frequenza di risonanza f1 in funzione di carico verticale (Mass) e
compressione (xb/lb). Produce heatmap 2D e superficie 3D.

Modifica i parametri nella sezione CONFIGURAZIONE SWEEP.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

import numpy as np
import matplotlib.pyplot as plt

from FLECSpy2.config import configFLECS
from FLECSpy2.sweep import sweep_load_compression_2d


# ===================== CONFIGURAZIONE SWEEP =====================
MASS_MIN = 0.1
MASS_MAX = 0.50
N_MASS = 200

COMP_MIN = 0.9505
COMP_MAX = 0.9755
N_COMP = 200

SHOW_HEATMAP = True
SHOW_SURFACE = True
# ===============================================================


def main():
    cfg = configFLECS()
    mass_vec = np.linspace(MASS_MIN, MASS_MAX, N_MASS)
    comp_vec = np.linspace(COMP_MIN, COMP_MAX, N_COMP)

    print(f'Grid {N_COMP} x {N_MASS} (compression x Mass)...')
    g = sweep_load_compression_2d(cfg, mass_vec, comp_vec,
                                  want_dynamics=True, verbose=True)
    F1 = g['F1']

    if SHOW_HEATMAP:
        plt.figure(figsize=(7.5, 5.5))
        # extent: [xmin,xmax,ymin,ymax]; origin lower per asse y crescente
        im = plt.imshow(F1, aspect='auto', origin='lower',
                        extent=[MASS_MIN, MASS_MAX, COMP_MIN * 100, COMP_MAX * 100],
                        cmap='viridis')
        cb = plt.colorbar(im)
        cb.set_label(r'$f_1$ [Hz]', fontsize=12)
        plt.xlabel('Mass [kg]', fontsize=13)
        plt.ylabel(r'$x_b/l_b$ [%]', fontsize=13)
        plt.title(r'$f_1$(Mass, $x_b$)', fontsize=14)
        plt.tight_layout()

    if SHOW_SURFACE:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        MM, CC = np.meshgrid(mass_vec, comp_vec * 100)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(MM, CC, F1, cmap='viridis',
                               edgecolor='none', antialiased=True)
        cb = fig.colorbar(surf, shrink=0.6, aspect=12)
        cb.set_label(r'$f_1$ [Hz]', fontsize=12)
        ax.set_xlabel('Mass [kg]', fontsize=12)
        ax.set_ylabel(r'$x_b/l_b$ [%]', fontsize=12)
        ax.set_zlabel(r'$f_1$ [Hz]', fontsize=12)
        ax.set_title(r'$f_1$(Mass, $x_b$) - 3D', fontsize=14)
        ax.view_init(elev=30, azim=-45)

    plt.show()


if __name__ == '__main__':
    main()