"""
tip_vs_load_vs_compression.py

Altezza della punta della lama (tip height) in funzione di carico verticale
(Mass) e compressione (xb/lb). Produce heatmap 2D e superficie 3D.

Gemello statico di frequency_vs_load_vs_compression.py: qui serve solo la
statica (l'altezza della punta esce dall'equilibrio), quindi si chiama lo
sweep con want_dynamics=False -> niente solve_dynamics, molto piu' veloce.

Modifica i parametri nella sezione CONFIGURAZIONE SWEEP.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt

from FLECSpy2.config import configFLECS
from FLECSpy2.sweep import sweep_load_compression_2d, save_matrix_csv


# ===================== CONFIGURAZIONE SWEEP =====================
MASS_MIN = 0.01
MASS_MAX = 0.1
N_MASS = 100

COMP_MIN = 0.89
COMP_MAX = 0.91
N_COMP = 100

SHOW_HEATMAP = True
SHOW_SURFACE = True
SAVE_CSV = None       # es. 'tip_megasweep.csv'; None = non salvare
# ===============================================================


def main():
    t_start = time.time()
    cfg = configFLECS()
    mass_vec = np.linspace(MASS_MIN, MASS_MAX, N_MASS)
    comp_vec = np.linspace(COMP_MIN, COMP_MAX, N_COMP)

    print(f'Grid {N_COMP} x {N_MASS} (compression x Mass)...')
    # want_dynamics=False: serve solo la statica (tip height) -> piu' veloce
    g = sweep_load_compression_2d(cfg, mass_vec, comp_vec,
                                  want_dynamics=False, verbose=True)
    HTIP = g['HTIP'] * 1e3      # da metri a mm, coerente con la figura

    if SAVE_CSV:
        save_matrix_csv(SAVE_CSV,
                        row_vals=comp_vec,   # righe = compressione xb/lb
                        col_vals=mass_vec,   # colonne = Mass [kg]
                        M=HTIP,              # celle = tip height [mm]
                        corner='xb_lb\\Mass_kg')
        print(f'CSV (matrice) salvato: {SAVE_CSV}  '
              f'({HTIP.shape[0]}x{HTIP.shape[1]}, celle in mm)')

    totalTime = time.time() - t_start
    print('')
    print('Total execution time: {:.3f} s'.format(totalTime))

    if SHOW_HEATMAP:
        plt.figure(figsize=(7.5, 5.5))
        # extent: [xmin,xmax,ymin,ymax]; origin lower per asse y crescente
        im = plt.imshow(HTIP, aspect='auto', origin='lower',
                        extent=[MASS_MIN, MASS_MAX, COMP_MIN * 100, COMP_MAX * 100],
                        cmap='viridis')
        cb = plt.colorbar(im)
        cb.set_label('Tip height [mm]', fontsize=12)
        plt.xlabel('Mass [kg]', fontsize=13)
        plt.ylabel(r'$x_b/l_b$ [%]', fontsize=13)
        plt.title(r'$h_{tip}$(Mass, $x_b$)', fontsize=14)
        plt.tight_layout()

    if SHOW_SURFACE:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        MM, CC = np.meshgrid(mass_vec, comp_vec * 100)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(MM, CC, HTIP, cmap='viridis',
                               edgecolor='none', antialiased=True)
        cb = fig.colorbar(surf, shrink=0.6, aspect=12)
        cb.set_label('Tip height [mm]', fontsize=12)
        ax.set_xlabel('Mass [kg]', fontsize=12)
        ax.set_ylabel(r'$x_b/l_b$ [%]', fontsize=12)
        ax.set_zlabel('Tip height [mm]', fontsize=12)
        ax.set_title(r'$h_{tip}$(Mass, $x_b$) - 3D', fontsize=14)
        ax.view_init(elev=30, azim=-45)

    plt.show()


if __name__ == '__main__':
    main()