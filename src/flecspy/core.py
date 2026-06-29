"""
flecs_core.py

Funzioni pure (senza stampa ne' plot) che incapsulano il calcolo di FLECS,
estratte da main.m / main.py.

Struttura:
    build_params(cfg)         -> costruisce la struct 'p' del sistema
    solve_equilibrium(p, cfg) -> STATICA: equilibrio e tutto cio' che ne deriva
    solve_dynamics(p, static, cfg) -> DINAMICA: matrici, riduzione, modi

Regole di design concordate:
  (1) build_params separata da solve_equilibrium: le mappe (Jd/Jm/cd/cm) e gli
      altri parametri NON dipendono dall'ottimizzatore ne' da xb, quindi si
      costruiscono una volta sola. Utile per 'find a working point', che varia
      xb e ri-risolve la statica senza ricostruire le mappe.
  (2) Confine statica/dinamica: la STATICA calcola G0, GX, HX, lambda (servono
      per lambda, momenti, ed essendo derivati del vincolo / gradiente del
      potenziale all'equilibrio sono quantita' statiche). La DINAMICA li RIUSA
      dal dict 'static' invece di ricalcolarli: evita ricalcoli e, soprattutto,
      garantisce che statica e dinamica usino esattamente lo stesso GX/HX/lambda
      valutati sullo stesso theta.

  G0 (gradiente del potenziale all'equilibrio) e' una quantita' statica e viene
  ottenuto direttamente da bladePotential(theta, p), SENZA costruire Kfull.
  Cosi' solve_equilibrium non dipende dalla matrice di rigidezza (dinamica).

Questi moduli non alterano alcun valore numerico rispetto a main.py: spostano
solo il codice. Il criterio di non-regressione e' che la suite pytest resti verde.
"""

import numpy as np
import sympy as sp
from scipy.optimize import minimize, NonlinearConstraint
from scipy.linalg import eigh

from flecspy.elements.cell_constants import cellConstants
from flecspy.elements.blade_maps import buildBladeMaps
from flecspy.elements.blade_potential import bladePotential
from flecspy.elements.blade_constraint import bladeConstraint
from flecspy.elements.build_mass_matrix import buildMassMatrix
from flecspy.elements.build_stiffness_matrix import buildStiffnessMatrix
from flecspy.elements.build_constraint_derivatives import buildConstraintDerivatives
from flecspy.elements.build_constraint_reduction import buildConstraintReduction
from flecspy.elements.blade_clamp_moments import bladeClampMoments


# =====================================================================
#  build_params
# =====================================================================
def build_params(cfg):
    """
    Costruisce la struct 'p' del sistema e theta0, replicando la sezione di
    main.m che va da cellConstants alla definizione di theta0.

    Restituisce:
        p      : dict dei parametri (mappe, dl, m, Kb, dtheta, gcoeff, flag, xb)
        theta0 : condizione iniziale geometrica (1-D)
        extra  : dict con quantita' di supporto utili a stampe/plot
                 (dtheta, dl, w, m, Kb, prf, costanti geometriche)
    """
    N = cfg['N']
    wb, hb, lb, rb = cfg['wb'], cfg['hb'], cfg['lb'], cfg['rb']
    ain, aout = cfg['ain'], cfg['aout']
    type_ = cfg['type']
    xb = cfg['xb']
    Eb, rhob = cfg['Eb'], cfg['rhob']

    # Profilo simbolico (come in main.m). cellConstants integra rispetto alla
    # variabile del profilo: riproduciamo il comportamento MATLAB osservato
    # (narea = psi - 0.327 psi^2) rinominando csi -> psi.
    csi = sp.symbols('csi')
    if type_ == 'tama':
        c1, c2, c3, beta = -0.377, 1.377, 0.195, 1.361
        prf = c1 + c2 * sp.cos(beta * csi) + c3 * sp.sin(beta * csi)
    elif type_ == 'triangle':
        prf = 1 - csi
    else:
        prf = sp.sympify(type_)
    psi = sp.symbols('psi')
    prf_psi = prf.subs(csi, psi)

    dtheta, dl, w, m, Kb = cellConstants(N, wb, lb, rb, hb, Eb, rhob, prf_psi, cfg['partition'])

    # Flag di configurazione
    constrainedTip = isinstance(aout, (int, float)) and not isinstance(aout, bool)
    constrainedLength = isinstance(xb, (int, float)) and not isinstance(xb, bool)

    p = {}
    p['ain'] = ain
    p['aout'] = aout
    p['dl'] = dl.reshape(-1, 1)
    p['Kb'] = Kb.reshape(-1, 1)
    p['dtheta'] = dtheta.reshape(-1, 1)
    p['m'] = m.reshape(-1, 1)
    p['Mass'] = cfg['Mass']
    p['g'] = cfg['g']
    p['nb'] = cfg['nb']
    p['constrainedTip'] = constrainedTip
    p['constrainedLength'] = constrainedLength
    p['xb'] = xb

    Jd, Jm, cd, cm = buildBladeMaps(N, constrainedTip, ain, aout)
    p['Jd'], p['Jm'], p['cd'], p['cm'] = Jd, Jm, cd, cm

    # gcoeff = flipud(cumsum(flipud(m))) - 0.5*m
    mflat = m.reshape(-1)
    p['gcoeff'] = (np.flip(np.cumsum(np.flip(mflat))) - 0.5 * mflat).reshape(-1, 1)

    # theta0 = ain + cumsum(dtheta(1:N)), poi tolto l'ultimo se constrainedTip
    theta0 = ain + np.cumsum(dtheta[:N])
    if constrainedTip:
        theta0 = theta0[:-1]
    theta0 = theta0.reshape(-1)

    extra = dict(dtheta=dtheta, dl=dl, w=w, m=m, Kb=Kb, prf=prf,
                 wb=wb, hb=hb, lb=lb, Eb=Eb, ain=ain, aout=aout, N=N)
    return p, theta0, extra


# =====================================================================
#  solve_equilibrium  (STATICA)
# =====================================================================
def solve_equilibrium(p, cfg, theta0=None):
    """
    Risolve la statica: trova l'equilibrio e calcola tutte le quantita' che
    ne derivano (profilo, stress, lambda, momenti, tip).

    La STATICA calcola G0, GX, HX, lambda (regola 2): G0 da bladePotential,
    GX/HX da buildConstraintDerivatives. La dinamica li riusera'.

    Inputs:
        p      : struct parametri (da build_params)
        cfg    : config (per N, neig, optimizer, hb, Eb, ...)
        theta0 : condizione iniziale; se None viene ricostruita da p

    Restituisce un dict 'static' con (tra gli altri):
        theta, theta_full, theta_m, x, y, xtip, htip, tipAngle,
        stress, smax, U0, ok, G0, GX, HX, lambda_, M_in, M_out
    """
    N = cfg['N']
    ain = p['ain']
    aout = p['aout']
    hb = cfg['hb']
    Eb = cfg['Eb']
    dl = p['dl'].reshape(-1)
    dtheta = p['dtheta'].reshape(-1)
    constrainedTip = p['constrainedTip']
    constrainedLength = p['constrainedLength']
    optimizer = cfg.get('optimizer', 'SLSQP')

    if theta0 is None:
        theta0 = ain + np.cumsum(dtheta[:N])
        if constrainedTip:
            theta0 = theta0[:-1]
        theta0 = theta0.reshape(-1)

    def Pot_fun(th):
        return bladePotential(th, p)[0]

    def Pot_grad(th):
        return np.asarray(bladePotential(th, p)[1], dtype=float).reshape(-1)

    # --- Minimizzazione ---
    if constrainedLength:
        def con_fun(th):
            _, ceq, _, _ = bladeConstraint(th, p)
            return ceq

        def con_jac_row(th):
            _, _, _, GCeq = bladeConstraint(th, p)
            return np.asarray(GCeq, dtype=float).reshape(1, -1)

        lwr = -np.pi / 2 * np.ones_like(theta0)
        upr = np.pi / 2 * np.ones_like(theta0)
        bounds = list(zip(lwr, upr))

        if optimizer == 'trust-constr':
            nlc = NonlinearConstraint(con_fun, 0.0, 0.0, jac=con_jac_row)
            res = minimize(Pot_fun, theta0, jac=Pot_grad, method='trust-constr',
                           bounds=bounds, constraints=[nlc],
                           options={'gtol': 1e-9, 'xtol': 1e-12,
                                    'maxiter': 100000, 'verbose': 0})
        else:
            con = {'type': 'eq', 'fun': con_fun,
                   'jac': lambda th: con_jac_row(th).reshape(-1)}
            res = minimize(Pot_fun, theta0, jac=Pot_grad, method='SLSQP',
                           bounds=bounds, constraints=[con],
                           options={'ftol': 1e-12, 'maxiter': 1000})
        theta = res.x
        U0 = res.fun
        ok = res.status
    else:
        res = minimize(Pot_fun, theta0, jac=Pot_grad, method='BFGS',
                       options={'gtol': 1e-9, 'maxiter': 100000})
        theta = res.x
        U0 = res.fun
        ok = res.status

    theta = np.asarray(theta, dtype=float).reshape(-1)

    # --- Profilo ---
    if constrainedTip:
        theta_full = np.concatenate([theta, [aout]])
        theta_m = (theta_full + np.concatenate([[ain], theta])) / 2
    else:
        theta_full = theta
        theta_m = (theta + np.concatenate([[ain], theta[:-1]])) / 2

    x = np.cumsum(dl * np.cos(theta_m))
    y = np.cumsum(dl * np.sin(theta_m))
    htip = y[-1]
    xtip = x[-1]
    tipAngle = aout if constrainedTip else theta[-1]

    # --- Stress ---
    if constrainedTip:
        theta_prev = np.concatenate([[ain], theta])
    else:
        theta_prev = np.concatenate([[ain], theta[:-1]])
    stress = np.abs(-hb / 2 * Eb / dl * (theta_full - theta_prev - dtheta))
    smax = float(np.max(np.abs(stress)))

    # --- G0 (gradiente del potenziale all'equilibrio): quantita' statica ---
    _, G0 = bladePotential(theta, p)
    G0 = np.asarray(G0, dtype=float).reshape(-1, 1)

    # --- Derivati del vincolo e lambda (statica) ---
    if constrainedLength:
        GX, HX, Xeq, _ = buildConstraintDerivatives(theta, p)
        GXc = np.asarray(GX, dtype=float).reshape(-1, 1)
        lambda_ = float((-(GXc.T @ G0) / (GXc.T @ GXc)).item())
    else:
        GX = np.empty((0, 1))
        HX = np.empty((0, 0))
        lambda_ = 0.0

    # --- Momenti di clamp (statica), solo se applicabile ---
    if constrainedTip and constrainedLength:
        M_in, M_out, _, _ = bladeClampMoments(theta, lambda_, p)
    else:
        M_in = M_out = None

    return dict(theta=theta, theta_full=theta_full, theta_m=theta_m,
                x=x, y=y, xtip=xtip, htip=htip, tipAngle=tipAngle,
                stress=stress, smax=smax, U0=U0, ok=ok,
                G0=G0, GX=GX, HX=HX, lambda_=lambda_,
                M_in=M_in, M_out=M_out, theta0=theta0)


# =====================================================================
#  solve_dynamics  (DINAMICA)
# =====================================================================
def solve_dynamics(p, static, cfg):
    """
    Calcola la dinamica attorno all'equilibrio: matrici di massa e rigidezza,
    riduzione del vincolo, problema agli autovalori (frequenze e modi).

    RIUSA dalla statica (regola 2): theta, lambda, GX, HX. NON li ricalcola.

    Inputs:
        p      : struct parametri
        static : dict restituito da solve_equilibrium
        cfg    : config (per neig)

    Restituisce un dict 'dynamics' con:
        Mtot, Kfull, Ktot, Mred, Kred, freqs, evals, modes (autovettori ridotti),
        Tred, keepIdx, J
    """
    neig = cfg['neig']
    constrainedLength = p['constrainedLength']

    theta = static['theta']
    lambda_ = static['lambda_']
    GX = static['GX']
    HX = static['HX']

    # Matrice di massa (dinamica)
    _, _, _, Mtot, _, _ = buildMassMatrix(theta, p)

    # Matrice di rigidezza (dinamica)
    Kfull, _, _ = buildStiffnessMatrix(theta, p)

    # Rigidezza vincolata: riusa lambda e HX dalla statica
    if constrainedLength:
        Ktot = Kfull + lambda_ * HX
    else:
        Ktot = Kfull

    # Riduzione del vincolo: riusa GX dalla statica
    if constrainedLength:
        J = int(np.argmax(np.abs(np.asarray(GX).reshape(-1))))
        Tred, keepIdx, _ = buildConstraintReduction(GX, J)
        Mred = Tred.T @ Mtot @ Tred
        Kred = Tred.T @ Ktot @ Tred
    else:
        J = None
        Tred = np.eye(Mtot.shape[0])
        keepIdx = np.arange(Mtot.shape[0])
        Mred = Mtot
        Kred = Ktot

    # Problema agli autovalori generalizzato (eigs 'smallestabs' -> eigh + sort)
    evals_all, Vr_all = eigh(Kred, Mred)
    order = np.argsort(np.abs(evals_all))
    sel = order[:neig]
    evals = np.real(evals_all[sel])
    modes = Vr_all[:, sel]
    #freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
    freqs = np.emath.sqrt(evals) / (2 * np.pi)

    return dict(Mtot=Mtot, Kfull=Kfull, Ktot=Ktot, Mred=Mred, Kred=Kred,
                freqs=freqs, evals=evals, modes=modes,
                Tred=Tred, keepIdx=keepIdx, J=J)
