"""
bladeClampMoments.py

Replica 1:1 di bladeClampMoments.m

Momenti di reazione al clamp (base) e al tip.
Validi solo quando constrainedTip = True e constrainedLength = True.

NOTE SULLA TRADUZIONE:
- Indici primo/ultimo elemento: MATLAB e1 = delta_th(1), eN = delta_th(N).
  -> Python: delta_th[0] e delta_th[N-1].
- I fattori 0.5, -1, +1 sono mantenuti HARDCODED come nell'originale
  (decisione esplicita: nessun refactor dalle mappe Jd/Jm in questa fase).
  Questi fattori corrispondono comunque a:
    -1, +1  = entrate di Jd nelle righe del clamp/tip (derivata di delta_th
              rispetto ad ain/aout)
    0.5     = entrate di Jm nelle stesse righe (derivata di theta_m rispetto
              ad ain/aout)
  ma li lasciamo come costanti letterali per fedelta' 1:1.

CONVENZIONI:
- theta_eq trattato come colonna (n,1).
- p deve contenere: constrainedTip, constrainedLength (bool), cd, Jd, cm, Jm,
  dtheta, Kb, gcoeff, dl, nb, g, Mass.
- Output: M_in, M_out, dU_dain, dU_daout (tutti float).

MATLAB '*' (scalari/elementi singoli) -> '*' in Python (sono scalari).
"""

import numpy as np


def bladeClampMoments(theta_eq, lambda_, p):
    """
    Replica 1:1 di bladeClampMoments.m

    Inputs:
        theta_eq : vettore equilibrio variabili libere (qualsiasi forma)
        lambda_  : moltiplicatore di Lagrange del vincolo orizzontale (scalare)
                   (nome 'lambda_' perche' 'lambda' e' keyword in Python)
        p        : dict parametri

    Outputs:
        M_in     : momento di reazione al clamp [N*m]
        M_out    : momento di reazione al tip   [N*m]
        dU_dain  : derivata parziale di U rispetto ad ain
        dU_daout : derivata parziale di U rispetto ad aout
    """
    # if ~p.constrainedTip, error
    if not p['constrainedTip']:
        raise ValueError('bladeClampMoments: requires constrainedTip = true')
    # if ~p.constrainedLength, error
    if not p['constrainedLength']:
        raise ValueError('bladeClampMoments: requires constrainedLength = true')

    # theta_eq = theta_eq(:)
    theta_eq = np.asarray(theta_eq, dtype=float).reshape(-1, 1)
    # N = length(p.dl)
    dl = np.asarray(p['dl'], dtype=float).reshape(-1, 1)
    N = dl.shape[0]

    cd = p['cd']
    Jd = p['Jd']
    cm = p['cm']
    Jm = p['Jm']
    dtheta = np.asarray(p['dtheta'], dtype=float).reshape(-1, 1)
    Kb = np.asarray(p['Kb'], dtype=float).reshape(-1, 1)
    gcoeff = np.asarray(p['gcoeff'], dtype=float).reshape(-1, 1)
    nb = p['nb']
    g = p['g']
    Mass = p['Mass']

    # Mappe all'equilibrio
    # delta_th = p.cd + p.Jd * theta_eq
    delta_th = cd + Jd @ theta_eq        # (N, 1)
    # theta_m  = p.cm + p.Jm * theta_eq
    theta_m = cm + Jm @ theta_eq         # (N, 1)

    # Strain elastici primo e ultimo elemento
    # e1 = delta_th(1) - p.dtheta(1)
    e1 = delta_th[0, 0] - dtheta[0, 0]
    # eN = delta_th(N) - p.dtheta(N)
    eN = delta_th[N - 1, 0] - dtheta[N - 1, 0]

    # dU/dain (solo elemento 1)
    # = nb*Kb(1)*e1*(-1) + nb*g*gcoeff(1)*dl(1)*cos(theta_m(1))*0.5 + Mass*g*dl(1)*cos(theta_m(1))*0.5
    dU_dain = nb * Kb[0, 0] * e1 * (-1) \
        + nb * g * gcoeff[0, 0] * dl[0, 0] * np.cos(theta_m[0, 0]) * 0.5 \
        + Mass * g * dl[0, 0] * np.cos(theta_m[0, 0]) * 0.5

    # dU/daout (solo elemento N)
    # = nb*Kb(N)*eN*(+1) + nb*g*gcoeff(N)*dl(N)*cos(theta_m(N))*0.5 + Mass*g*dl(N)*cos(theta_m(N))*0.5
    dU_daout = nb * Kb[N - 1, 0] * eN * (+1) \
        + nb * g * gcoeff[N - 1, 0] * dl[N - 1, 0] * np.cos(theta_m[N - 1, 0]) * 0.5 \
        + Mass * g * dl[N - 1, 0] * np.cos(theta_m[N - 1, 0]) * 0.5

    # dX/dain e dX/daout
    # dX_dain  = -dl(1) * sin(theta_m(1)) * 0.5
    dX_dain = -dl[0, 0] * np.sin(theta_m[0, 0]) * 0.5
    # dX_daout = -dl(N) * sin(theta_m(N)) * 0.5
    dX_daout = -dl[N - 1, 0] * np.sin(theta_m[N - 1, 0]) * 0.5

    # Momenti di reazione
    # M_in  = -(dU_dain  + lambda * dX_dain)
    M_in = -(dU_dain + lambda_ * dX_dain)
    # M_out = -(dU_daout + lambda * dX_daout)
    M_out = -(dU_daout + lambda_ * dX_daout)

    return M_in, M_out, dU_dain, dU_daout
