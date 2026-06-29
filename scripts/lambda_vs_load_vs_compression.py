"""

Moltiplicatore di Lagrange lambda del vincolo orizzontale, in funzione di
carico verticale (Mass) e compressione (xb/lb). Heatmap 2D e superficie 3D,
con evidenziata la curva lambda = 0.

lambda e' definito da:  grad U + lambda * grad X = 0  (vedi flecs_core).
Fisicamente, il segno di lambda separa regimi diversi del vincolo; la curva
lambda = 0 e' il luogo critico dove il contributo del vincolo alla rigidezza
cambia segno.

"""
import sys, os


import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from flecspy.config import configFLECS
from flecspy.sweep import sweep_load_compression_2d


# ===================== CONFIGURAZIONE SWEEP =====================
MASS_MIN = 0.30
MASS_MAX = 0.50
N_MASS = 40

COMP_MIN = 0.9705
COMP_MAX = 0.9755
N_COMP = 40

SHOW_HEATMAP = True
SHOW_SURFACE = True
# ===============================================================


def main():
    t_start = time.time()
    cfg = configFLECS()
    mass_vec = np.linspace(MASS_MIN, MASS_MAX, N_MASS)
    comp_vec = np.linspace(COMP_MIN, COMP_MAX, N_COMP)

    print(f'Grid {N_COMP} x {N_MASS} (compression x Mass)...')
    g = sweep_load_compression_2d(cfg, mass_vec, comp_vec,
                                  want_dynamics=False, verbose=True)
    LAM = g['LAMBDA']
    
    totalTime = time.time() - t_start
    print('')
    print('Total execution time: {:.3f} s'.format(totalTime))  
    
    # Scelta della colormap in base ai dati:
    #  - se lambda ATTRAVERSA lo zero (sia valori >0 sia <0), usa una colormap
    #    divergente centrata su zero (blu<0, bianco=0, rosso>0) e disegna la
    #    curva critica lambda=0;
    #  - se lambda e' tutto di un segno, usa una colormap sequenziale sul range
    #    REALE dei dati, senza forzare lo zero (altrimenti meta' scala resta
    #    inutilizzata e i colori risultano compressi).
    finite = LAM[np.isfinite(LAM)]
    crosses_zero = finite.size > 0 and finite.min() < 0 < finite.max()

    if finite.size:
        lo = np.percentile(finite, 2)
        hi = np.percentile(finite, 98)
    else:
        lo, hi = 0.0, 1.0

    if crosses_zero:
        vmax = max(abs(lo), abs(hi))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = 'RdBu_r'
    else:
        # tutto positivo (o tutto negativo): sequenziale sul range effettivo
        norm = None
        cmap = 'viridis'
        vmin_seq, vmax_seq = lo, hi

    if SHOW_HEATMAP:
        plt.figure(figsize=(7.5, 5.5))
        if crosses_zero:
            im = plt.imshow(LAM, aspect='auto', origin='lower',
                            extent=[MASS_MIN, MASS_MAX, COMP_MIN * 100, COMP_MAX * 100],
                            cmap=cmap, norm=norm)
        else:
            im = plt.imshow(LAM, aspect='auto', origin='lower',
                            extent=[MASS_MIN, MASS_MAX, COMP_MIN * 100, COMP_MAX * 100],
                            cmap=cmap, vmin=vmin_seq, vmax=vmax_seq)
        cb = plt.colorbar(im)
        cb.set_label(r'$\lambda$ [N]', fontsize=12)
        # curva lambda = 0 solo se i dati la attraversano davvero
        if crosses_zero:
            MM, CC = np.meshgrid(mass_vec, comp_vec * 100)
            try:
                cs = plt.contour(MM, CC, LAM, levels=[0.0], colors='k',
                                linewidths=2.5)
                plt.clabel(cs, fmt=r'$\lambda=0$', fontsize=10)
            except Exception:
                pass
        plt.xlabel('Mass [kg]', fontsize=13)
        plt.ylabel(r'$x_b/l_b$ [%]', fontsize=13)
        plt.title(r'$\lambda$(Mass, $x_b$)', fontsize=14)
        plt.tight_layout()

    if SHOW_SURFACE:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        MM, CC = np.meshgrid(mass_vec, comp_vec * 100)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        if crosses_zero:
            surf = ax.plot_surface(MM, CC, LAM, cmap=cmap, norm=norm,
                                   edgecolor='none', antialiased=True)
        else:
            surf = ax.plot_surface(MM, CC, LAM, cmap=cmap,
                                   vmin=vmin_seq, vmax=vmax_seq,
                                   edgecolor='none', antialiased=True)
        cb = fig.colorbar(surf, shrink=0.6, aspect=12)
        cb.set_label(r'$\lambda$ [N]', fontsize=12)
        ax.set_xlabel('Mass [kg]', fontsize=12)
        ax.set_ylabel(r'$x_b/l_b$ [%]', fontsize=12)
        ax.set_zlabel(r'$\lambda$ [N]', fontsize=12)
        ax.set_title(r'$\lambda$(Mass, $x_b$) - 3D', fontsize=14)
        ax.view_init(elev=30, azim=-45)

    plt.show()


if __name__ == '__main__':
    main()