"""
bladePotential.py

Replica 1:1 di bladePotential.m

Energia potenziale della lama (elastica + gravitazionale di lama e tip)
e relativo gradiente analitico rispetto alle variabili libere theta.

CONVENZIONI:
- theta in ingresso puo' essere 1-D o 2-D; il gradiente viene restituito con
  la STESSA forma dell'input (replica di reshape(G, size(theta_in)) in MATLAB).
- Internamente si lavora con theta come vettore colonna (n,1), come 'theta(:)'.
- I campi di p attesi (vettori colonna (n_*,1) o (N,1) come da main):
    p['cd'], p['Jd'], p['cm'], p['Jm']   (mappe affini)
    p['dl']      (N,1)   lunghezze elementi
    p['dtheta']  (N,1)   incrementi angolari a riposo
    p['Kb']      (N,1)   costanti elastiche
    p['gcoeff']  (N,1)   coefficienti gravitazionali (massa cumulata - 0.5 m)
    p['nb']      scalare numero lame
    p['g']       scalare gravita'
    p['Mass']    scalare massa di tip

MATLAB '.*' -> NumPy '*' (element-wise)
MATLAB '*'  -> NumPy '@' (matriciale)
MATLAB ".'" -> NumPy '.T' (trasposta non coniugata)
"""

import numpy as np


def bladePotential(theta, p):
    """
    Replica 1:1 di bladePotential.m

    Inputs:
        theta : ndarray (free angles). Forma arbitraria (1-D o 2-D).
        p     : dict dei parametri (vedi docstring modulo).

    Outputs:
        U : float, energia potenziale totale.
        G : ndarray, gradiente di U rispetto a theta, STESSA forma di theta.
            (restituito solo se richiesto; qui sempre calcolato e ritornato come
             secondo valore -> il chiamante puo' ignorarlo)

    NOTA: in MATLAB il gradiente e' calcolato solo se nargout > 1. In Python
    non esiste nargout; restituiamo sempre (U, G). I chiamanti che vogliono
    solo U useranno bladePotential(...)[0]. Questo NON cambia i valori, solo
    il fatto che G viene sempre calcolato.
    """
    theta_in = theta
    # theta = theta(:)  -> colonna
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)

    cd = p['cd']
    Jd = p['Jd']
    cm = p['cm']
    Jm = p['Jm']

    # Affine maps
    # delta_th = p.cd + p.Jd * theta
    delta_th = cd + Jd @ theta
    # theta_m  = p.cm + p.Jm * theta
    theta_m = cm + Jm @ theta

    dl = p['dl']
    dtheta = p['dtheta']
    Kb = p['Kb']
    gcoeff = p['gcoeff']
    nb = p['nb']
    g = p['g']
    Mass = p['Mass']

    # Vertical increments
    # dy = p.dl .* sin(theta_m)
    dy = dl * np.sin(theta_m)

    # Potential energy
    # e = delta_th - p.dtheta
    e = delta_th - dtheta

    # Ue = 0.5 * p.nb * sum(p.Kb .* e.^2)
    Ue = 0.5 * nb * np.sum(Kb * e**2)
    # Ug_blade = p.nb * p.g * sum(p.gcoeff .* dy)
    Ug_blade = nb * g * np.sum(gcoeff * dy)
    # Ug_tip = p.Mass * p.g * sum(dy)
    Ug_tip = Mass * g * np.sum(dy)

    # U = Ue + Ug_blade + Ug_tip
    U = float(Ue + Ug_blade + Ug_tip)

    # Gradiente (in MATLAB solo se nargout>1; qui sempre)
    # Elastic part: Ge = p.nb * (p.Jd.' * (p.Kb .* e))
    Ge = nb * (Jd.T @ (Kb * e))
    # Gravity of blade: Gg_blade = p.nb * p.g * (p.Jm.' * (p.gcoeff .* p.dl .* cos(theta_m)))
    Gg_blade = nb * g * (Jm.T @ (gcoeff * dl * np.cos(theta_m)))
    # Gravity of tip mass: Gg_tip = p.Mass * p.g * (p.Jm.' * (p.dl .* cos(theta_m)))
    Gg_tip = Mass * g * (Jm.T @ (dl * np.cos(theta_m)))

    # G = Ge + Gg_blade + Gg_tip
    G = Ge + Gg_blade + Gg_tip

    # Return gradient with same shape as input: reshape(G, size(theta_in))
    G = G.reshape(np.shape(theta_in))

    return U, G
