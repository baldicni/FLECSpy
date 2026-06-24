"""
buildConstraintDerivatives.py

Replica 1:1 di buildConstraintDerivatives.m

Gradiente e Hessiano del vincolo orizzontale.

Vincolo:
    X(theta) = sum_i dl_i * cos(theta_m,i) = xb
dove:
    theta_m = cm + Jm * theta

PUNTI DELLA TRADUZIONE:
- GX = p.Jm.' * (-p.dl .* sin(theta_m))   -> Jm.T @ (-dl * sin(theta_m))
- HX = p.Jm.' * diag(-p.dl .* cos(theta_m)) * p.Jm
       -> Jm.T @ diag((-dl*cos(theta_m)).ravel()) @ Jm
  Nota: np.diag richiede un vettore 1-D, quindi appiattisco l'argomento.

CONVENZIONI:
- theta_eq trattato come colonna (n,1).
- p['dl'] (N,1), p['Jm'] (N,nvar), p['cm'] (N,1).
- Output: GX (nvar,1), HX (nvar,nvar), Xeq float, theta_m (N,1).

MATLAB '.*' -> '*'(elementwise), '*' -> '@'(matriciale), ".'" -> '.T'
"""

import numpy as np


def buildConstraintDerivatives(theta_eq, p):
    """
    Replica 1:1 di buildConstraintDerivatives.m

    Inputs:
        theta_eq : equilibrium free-angle vector (qualsiasi forma; usato come colonna)
        p        : dict con dl, Jm, cm

    Outputs:
        GX      : (nvar, 1) gradiente di X rispetto agli angoli liberi
        HX      : (nvar, nvar) Hessiano di X rispetto agli angoli liberi
        Xeq     : float, proiezione orizzontale in theta_eq
        theta_m : (N, 1) angoli medi di elemento in theta_eq
    """
    # theta_eq = theta_eq(:)
    theta_eq = np.asarray(theta_eq, dtype=float).reshape(-1, 1)

    dl = np.asarray(p['dl'], dtype=float).reshape(-1, 1)  # (N,1)
    Jm = p['Jm']                                          # (N, nvar)
    cm = p['cm']                                          # (N, 1)

    # Mean element angles: theta_m = p.cm + p.Jm * theta_eq
    theta_m = cm + Jm @ theta_eq        # (N, 1)

    # Horizontal projection: dx = p.dl .* cos(theta_m); Xeq = sum(dx)
    dx = dl * np.cos(theta_m)           # (N, 1)
    Xeq = float(np.sum(dx))

    # Gradient of X(theta): GX = p.Jm.' * (-p.dl .* sin(theta_m))
    GX = Jm.T @ (-dl * np.sin(theta_m))     # (nvar, 1)

    # Hessian of X(theta): HX = p.Jm.' * diag(-p.dl .* cos(theta_m)) * p.Jm
    diag_arg = (-dl * np.cos(theta_m)).reshape(-1)   # (N,)
    HX = Jm.T @ np.diag(diag_arg) @ Jm               # (nvar, nvar)

    return GX, HX, Xeq, theta_m
