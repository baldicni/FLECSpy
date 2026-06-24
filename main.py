"""
main.py

Thin wrapper di FLECS: chiama le funzioni pure di flecs_core, stampa i
risultati (come i 'disp' di main.m) e, se richiesto, disegna i grafici
chiamando flecs_plots.

Il calcolo vive in flecs_core (build_params / solve_equilibrium /
solve_dynamics); qui c'e' solo orchestrazione, stampa e (opzionale) plot.

Il flag plotOut e' superfluo nella nuova struttura: i plot sono funzioni
separate che il chiamante invoca se vuole. Per compatibilita' con la config
esistente, se cfg['plotOut'] e' True i grafici vengono mostrati.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  

import time
import numpy as np

from FLECSpy2.config import configFLECS
from FLECSpy2.core import build_params, solve_equilibrium, solve_dynamics


def main():
    t_start = time.time()
    cfg = configFLECS()

    # --- Parametri del sistema ---
    p, theta0, extra = build_params(cfg)

    # --- STATICA ---

    static = solve_equilibrium(p, cfg, theta0)
    print(' ')
    print('================= Static ================= ')
    print(' ')
    print('Rest values:')
    print('Tip angle = {:.3G} pi'.format(static['tipAngle'] / np.pi))
    print('Tip height = {:.3G} m'.format(static['htip']))
    print('Tip abscissa = {:.3G} m'.format(static['xtip']))
    print('Max stress = {:.3G} Pa'.format(round(static['smax'])))

    if p['constrainedLength']:
        print('Estimated lambda = {:.3G} N'.format(static['lambda_']))

    if p['constrainedTip'] and p['constrainedLength']:
        print(' ')
        print('Clamp moments:')
        print('M_in  (base) = {:.3G} N*m'.format(static['M_in']))
        print('M_out (tip)   = {:.3G} N*m'.format(static['M_out']))

    # --- DINAMICA ---
    dynamics = solve_dynamics(p, static, cfg)
    print(' ')
    print('================= Dynamics ================= ')
    print('')
    print('Modes:')
    for i in range(cfg['neig']):
        print('f_{} = {:.3G} Hz'.format(i + 1, dynamics['freqs'][i]))

    totalTime = time.time() - t_start
    print('')
    print('Total execution time: {:.3f} s'.format(totalTime))    

    # --- PLOT ---
    from FLECSpy2.plots import (plot_blade_shape, plot_profile,plot_stress, plot_modes)
    import matplotlib.pyplot as plt        

    plot_blade_shape(p, extra)
    plot_profile(static)
    plot_stress(static, p, cfg)
    plot_modes(static, dynamics, p, cfg)
    plt.show()



    # Dizionario unificato dei risultati (retrocompatibile con la vecchia main:
    # espone le stesse chiavi che la suite usa).
    return dict(
        theta=static['theta'], U0=static['U0'], lambda_=static['lambda_'],
        freqs=dynamics['freqs'],
        Mtot=dynamics['Mtot'], Ktot=dynamics['Ktot'], Kfull=dynamics['Kfull'],
        GX=static['GX'], HX=static['HX'],
        Mred=dynamics['Mred'], Kred=dynamics['Kred'],
        M_in=static['M_in'], M_out=static['M_out'], J=dynamics['J'],
        static=static, dynamics=dynamics,
    )


if __name__ == '__main__':
    main()
