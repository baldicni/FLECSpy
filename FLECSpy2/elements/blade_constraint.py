"""
bladeConstraint.py

Replica 1:1 di bladeConstraint.m

Funzione di vincolo nel formato MATLAB fmincon:
    [c, ceq, GC, GCeq] = bladeConstraint(theta, p)

dove:
    c    : vincoli di disuguaglianza (qui vuoto)
    ceq  : vincoli di uguaglianza   -> X(theta) - xb
    GC   : gradiente dei vincoli di disuguaglianza (qui vuoto)
    GCeq : gradiente dei vincoli di uguaglianza     -> Jm.' * (-dl .* sin(theta_m))

NOTA SULL'ADATTAMENTO A SciPy (rimandato a main.py):
  fmincon vuole [c, ceq, GC, GCeq] e per i gradienti usa la convenzione
  "colonne = vincoli" (GCeq e' nvar x n_ceq). Qui n_ceq = 1, quindi GCeq e'
  (nvar, 1). scipy.optimize userà invece un dict {'type':'eq','fun':...,'jac':...}
  dove 'fun' restituisce ceq come scalare/array e 'jac' restituisce il gradiente
  come riga (1, nvar) o (nvar,). L'adattamento sara' fatto in main.py; questo
  modulo resta fedele alla firma MATLAB.

In MATLAB i gradienti sono calcolati solo se nargout > 2. In Python non esiste
nargout: restituiamo sempre i quattro valori (c, ceq, GC, GCeq). I valori non
cambiano; cambia solo che GC/GCeq sono sempre calcolati.

CONVENZIONI:
- theta trattato come colonna (n,1).
- c = [] (lista vuota, come MATLAB []), GC = [] (idem).
- ceq scalare (float).
- GCeq (nvar, 1).
"""

import numpy as np


def bladeConstraint(theta, p):
    """
    Replica 1:1 di bladeConstraint.m

    Inputs:
        theta : free angles (qualsiasi forma; usato come colonna)
        p     : dict con cm, Jm, dl, xb

    Outputs:
        c    : np.empty((0,1)) -- vincoli di disuguaglianza assenti
        ceq  : float -- X(theta) - xb
        GC   : np.empty((0,0)) -- gradiente disuguaglianze assente
        GCeq : (nvar, 1) -- gradiente del vincolo di uguaglianza
    """
    # theta = theta(:)
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)

    cm = p['cm']
    Jm = p['Jm']
    dl = np.asarray(p['dl'], dtype=float).reshape(-1, 1)
    xb = p['xb']

    # Mean element angles: theta_m = p.cm + p.Jm * theta
    theta_m = cm + Jm @ theta            # (N, 1)

    # Total horizontal projection: dx = p.dl .* cos(theta_m); X = sum(dx)
    dx = dl * np.cos(theta_m)
    X = float(np.sum(dx))

    # c = []; ceq = X - p.xb
    c = np.empty((0, 1))
    ceq = X - xb

    # Gradienti (in MATLAB solo se nargout>2; qui sempre)
    # GC = []
    GC = np.empty((0, 0))
    # GCeq = p.Jm.' * (-p.dl .* sin(theta_m))   % nvar x 1
    GCeq = Jm.T @ (-dl * np.sin(theta_m))        # (nvar, 1)

    return c, ceq, GC, GCeq
