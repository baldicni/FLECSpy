"""

Altezza della punta della lama (tip height) in funzione del carico verticale
(Mass), a compressione fissa. Supporta piu' compressioni.

Solo statica: non serve calcolare la dinamica, quindi e' veloce.

"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

import time
import numpy as np
import matplotlib.pyplot as plt

from flecspy.config import configFLECS
from flecspy.sweep import sweep_load_1d, save_csv


# ===================== CONFIGURAZIONE SWEEP =====================
MASS_MIN = 0.38        # [kg]
MASS_MAX = 0.42        # [kg]
N_MASS = 80

# Una o piu' compressioni xb/lb.
COMPRESSIONS = [0.9706]

SAVE_CSV = None        # es. 'tip_vs_load.csv'; None = non salvare
# ===============================================================


def main():
    t_start = time.time()
    cfg = configFLECS()
    mass_vec = np.linspace(MASS_MIN, MASS_MAX, N_MASS)

    plt.figure(figsize=(7.5, 5))

    for comp in COMPRESSIONS:
        # want_dynamics=False: serve solo la statica (tip height) -> piu' veloce
        res = sweep_load_1d(cfg, mass_vec, compression=comp,
                            want_dynamics=False, verbose=True)
        ok = res['converged']
        label = f'$x_b/l_b$ = {comp:.4f}'
        plt.plot(res['mass'][ok], res['htip'][ok] * 1e3, 'o-',
                 linewidth=2, markersize=4, label=label)
        print(f'compression {comp:.4f}: {int(ok.sum())}/{len(mass_vec)} convergent')

        if SAVE_CSV and comp == COMPRESSIONS[0]:
            save_csv(SAVE_CSV, [res['mass'], res['htip'] * 1e3],
                     ['Mass_kg', 'htip_mm'])

    if len(COMPRESSIONS) > 1:
        plt.legend(fontsize=11, title='Compression')

    totalTime = time.time() - t_start
    print('')
    print('Total execution time: {:.3f} s'.format(totalTime))  


    plt.xlabel('Vertical load [kg]', fontsize=13)
    plt.ylabel('Tip height [mm]', fontsize=13)
    plt.title('Tip height vs vertical load - Stability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.box(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()