"""
buildMassMatrix.py

Replica 1:1 di buildMassMatrix.m

Costruisce la matrice di massa in coordinate di angolo libero.

PUNTI DELICATI DELLA TRADUZIONE:
- mu_cum = flipud(cumsum(flipud(m))) - 0.5*m   (massa cumulata 'sotto' l'elemento)
  -> np.flip(np.cumsum(np.flip(m))) - 0.5*m
- Doppio loop con mu_cum(max(i,j)): in MATLAB i,j sono 1-based e max(i,j) e'
  l'indice 1-based piu' grande. In Python i,j sono 0-based, quindi
  max(i,j) (0-based) corrisponde allo STESSO elemento fisico: l'elemento di
  indice piu' alto. La traduzione e' diretta: mu_cum[max(i,j)] con i,j 0-based.
- bx_mid = (-dl .* sin(theta_m)).'  -> riga (1 x N)
  bx = bx_mid * p.Jm                -> riga (1 x nvar)
  Mtip = p.Mass * (bx.' * bx + by.' * by)  -> outer product, (nvar x nvar)

CONVENZIONI:
- theta_eq trattato come colonna (n,1) via reshape(-1,1).
- p['dl'], p['m'] attesi come array (N,) o (N,1); qui li appiattisco a 1-D
  per i calcoli scalari e li ridimensiono dove serve il prodotto matriciale.
- p['Jm'], p['cm'] come da buildBladeMaps: (N, nvar) e (N, 1).
- Output Mmid (N,N), Mblade/Mtip/Mtot (nvar,nvar), mu_cum (N,), theta_m (N,1).

MATLAB '.*' -> '*'(elementwise), '*' -> '@'(matriciale), ".'" -> '.T'
"""

import numpy as np


def buildMassMatrix(theta_eq, p):
    """
    Replica 1:1 di buildMassMatrix.m

    Inputs:
        theta_eq : free variables (qualsiasi forma; usato come colonna)
        p        : dict con cm, Jm, dl, m, nb, Mass

    Outputs:
        Mmid      : (N, N)        matrice di massa in coord. angolari di midpoint
        Mblade    : (nvar, nvar)  contributo lama in coord. libere
        Mtip      : (nvar, nvar)  contributo massa di tip
        Mtot      : (nvar, nvar)  matrice di massa totale
        mu_cum    : (N,)          massa cumulata - 0.5 m
        theta_m   : (N, 1)        angoli medi di elemento all'equilibrio
    """
    # theta_eq = theta_eq(:)  -> colonna
    theta_eq = np.asarray(theta_eq, dtype=float).reshape(-1, 1)

    dl = np.asarray(p['dl'], dtype=float).reshape(-1, 1)   # (N,1)
    m = np.asarray(p['m'], dtype=float).reshape(-1)        # (N,) per mu_cum
    Jm = p['Jm']                                           # (N, nvar)
    cm = p['cm']                                           # (N, 1)
    nb = p['nb']
    Mass = p['Mass']

    N = dl.shape[0]   # length(p.dl)

    # Mean element angles at equilibrium: theta_m = p.cm + p.Jm * theta_eq
    theta_m = cm + Jm @ theta_eq          # (N, 1)

    # Cumulative mass below each element:
    # mu_cum = flipud(cumsum(flipud(m))) - 0.5*m
    mu_cum = np.flip(np.cumsum(np.flip(m))) - 0.5 * m     # (N,)

    # Mass matrix in element-midpoint angular coordinates
    # Mmid(i,j) = nb * dl(i)*dl(j) * mu_cum(max(i,j)) * cos(theta_m(i)-theta_m(j))
    dl_flat = dl.reshape(-1)              # (N,)
    tm_flat = theta_m.reshape(-1)         # (N,)
    Mmid = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Mmid[i, j] = nb * dl_flat[i] * dl_flat[j] * mu_cum[max(i, j)] * \
                         np.cos(tm_flat[i] - tm_flat[j])

    # Map blade mass matrix to free-angle coordinates
    # Mblade = p.Jm.' * Mmid * p.Jm
    Mblade = Jm.T @ Mmid @ Jm

    # Tip-mass contribution in free-angle coordinates
    # bx_mid = (-dl .* sin(theta_m)).'   -> (1, N)
    # by_mid = ( dl .* cos(theta_m)).'   -> (1, N)
    bx_mid = (-dl * np.sin(theta_m)).T    # (1, N)
    by_mid = (dl * np.cos(theta_m)).T     # (1, N)

    # bx = bx_mid * p.Jm   -> (1, nvar)
    # by = by_mid * p.Jm   -> (1, nvar)
    bx = bx_mid @ Jm      # (1, nvar)
    by = by_mid @ Jm      # (1, nvar)

    # Mtip = p.Mass * (bx.' * bx + by.' * by)  -> (nvar, nvar)
    Mtip = Mass * (bx.T @ bx + by.T @ by)

    # Total mass matrix in free-angle coordinates
    Mtot = Mblade + Mtip

    return Mmid, Mblade, Mtip, Mtot, mu_cum, theta_m
