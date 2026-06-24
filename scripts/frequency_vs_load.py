"""
sweep_freq_vs_load.py

Prima frequenza di risonanza f1 in funzione del carico verticale (Mass),
a compressione fissa. Supporta piu' compressioni (piu' curve sullo stesso
grafico) passando una lista in COMPRESSIONS.

Uso:
    python sweep_freq_vs_load.py
Modifica i parametri nella sezione CONFIGURAZIONE SWEEP qui sotto.

I dati grezzi possono essere salvati in CSV impostando SAVE_CSV.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

import numpy as np
import matplotlib.pyplot as plt

from FLECSpy2.config  import configFLECS
from FLECSpy2.sweep import sweep_load_1d, save_csv


# ===================== CONFIGURAZIONE SWEEP =====================
MASS_MIN = 0.25        # [kg]
MASS_MAX = 0.50        # [kg]
N_MASS = 80

# Una o piu' compressioni xb/lb. Per una sola curva: [0.9740].
# Per piu' curve: [0.9710, 0.9725, 0.9740].
COMPRESSIONS = [0.9710]

COLOR_BY_STRESS = True     # se True (e una sola compressione) colora per stress
SAVE_CSV = None            # es. 'freq_vs_load.csv'; None = non salvare
# ===============================================================


def main():
    cfg = configFLECS()
    mass_vec = np.linspace(MASS_MIN, MASS_MAX, N_MASS)

    plt.figure(figsize=(7.5, 5))

    if len(COMPRESSIONS) == 1 and COLOR_BY_STRESS:
        # Curva singola colorata per stress massimo (come l'esempio MATLAB,
        # ma con il colore mappato sulla linea via LineCollection).
        comp = COMPRESSIONS[0]
        res = sweep_load_1d(cfg, mass_vec, compression=comp,
                            want_dynamics=True, verbose=True)
        m, f1, smax = res['mass'], res['f1'], res['smax']
        ok = res['converged']

        from matplotlib.collections import LineCollection
        pts = np.array([m[ok], f1[ok]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        smid = 0.5 * (smax[ok][:-1] + smax[ok][1:]) / 1e6
        lc = LineCollection(segs, cmap='viridis', linewidth=2.5)
        lc.set_array(smid)
        plt.gca().add_collection(lc)
        plt.scatter(m[ok], f1[ok], c=smax[ok] / 1e6, cmap='viridis',
                    s=25, zorder=3, edgecolor='none')
        cb = plt.colorbar(lc)
        cb.set_label(r'$\sigma_{max}$ [MPa]', fontsize=12)
        plt.autoscale()

        if SAVE_CSV:
            save_csv(SAVE_CSV, [m, f1, smax / 1e6],
                     ['Mass_kg', 'f1_Hz', 'MaxStress_MPa'])
    else:
        # Piu' compressioni: una curva per ciascuna.
        for comp in COMPRESSIONS:
            res = sweep_load_1d(cfg, mass_vec, compression=comp,
                                want_dynamics=True, verbose=False)
            ok = res['converged']
            plt.plot(res['mass'][ok], res['f1'][ok], '-', linewidth=2,
                     label=f'$x_b/l_b$ = {comp:.4f}')
            print(f'compression {comp:.4f}: {int(ok.sum())}/{len(mass_vec)} convergenti')
        plt.legend(fontsize=11, title='Compression')

        if SAVE_CSV:
            # salva la prima compressione come riferimento
            res0 = sweep_load_1d(cfg, mass_vec, compression=COMPRESSIONS[0],
                                 want_dynamics=True)
            save_csv(SAVE_CSV, [res0['mass'], res0['f1']],
                     ['Mass_kg', 'f1_Hz'])

    plt.xlabel('Vertical load [kg]', fontsize=13)
    plt.ylabel(r'Frequency [Hz]', fontsize=13)
    plt.title('Resonant frequency vs vertical load', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.box(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()