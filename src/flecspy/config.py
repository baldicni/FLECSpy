"""
configFLECS.py

Replica 1:1 di configFLECS.m

Restituisce un dizionario `cfg` con gli stessi campi e valori del file MATLAB.
La struttura `cfg.<campo>` di MATLAB diventa qui `cfg['<campo>']`.

I campi che in MATLAB possono essere 'none' o numerici (aout, xb) sono mantenuti
con la stessa semantica: o una stringa 'none', o un valore numerico (float).
"""

import numpy as np


def configFLECS():
    """Configuration file for FLECS. Replica 1:1 di configFLECS.m."""
    cfg = {}

    # %% Shape
    cfg['wb'] = 49e-3            # Blade base width [m]
    cfg['hb'] = 0.35e-3          # Blade thickness [m]
    cfg['lb'] = 52e-3            # Blade length [m]
    cfg['rb'] = 1e9              # Blade rest curvature [m]
    cfg['ain'] = np.pi / 9       # Clamp angle [rad]
    cfg['aout'] = -np.pi / 9     # Tip angle ['none' or numeric]
    cfg['type'] = '1-0.654*csi'  # Profile law
    cfg['xb'] = 0.9702 * cfg['lb']  # Horizontal projection ['none' or numeric]

    # %% Material
    cfg['Eb'] = 210e9            # Young modulus [Pa]
    cfg['UTSb'] = 1.4e9          # UTS [Pa]
    cfg['yldb'] = 1.15e9         # Yield [Pa]
    cfg['rhob'] = 7.85e3         # Density [kg/m^3]

    # %% Environment
    cfg['g'] = 9.81              # Gravity [m/s^2]
    cfg['nb'] = 1                # Number of blades
    cfg['Mass'] = 0.41           # Tip mass [kg]

    # %% Algorithm
    cfg['N'] = 64                # Number of blade partitions
    cfg['neig'] = 6             # Number of modes
    cfg['partition'] = 'length'  # 'length' or 'area'
    cfg['optimizer'] = 'SLSQP'   # 'SLSQP' (default, fast) or 'trust-constr' (slower, more accurate on theta)

    return cfg
