"""
buildStiffnessMatrix.py

Hessiano dell'energia potenziale rispetto alle variabili libere theta.

VERSIONE ANALITICA (sostituisce le differenze finite).

MOTIVAZIONE
-----------
L'energia potenziale di bladePotential e' esattamente

    U(theta) = 0.5*nb*sum( Kb .* e.^2 )
             + nb*g*sum( gcoeff .* dl .* sin(theta_m) )
             + Mass*g*sum( dl .* sin(theta_m) )

con le mappe AFFINI

    delta_th = cd + Jd @ theta        e = delta_th - dtheta
    theta_m  = cm + Jm @ theta

Essendo le mappe affini, l'Hessiano si ottiene in forma chiusa derivando
il gradiente analitico gia' restituito da bladePotential:

    d/dtheta [ nb * Jd' (Kb .* e) ]            =  nb * Jd' diag(Kb) Jd
    d/dtheta [ nb*g * Jm' (gcoeff.*dl.*cos) ]  = -nb*g * Jm' diag(gcoeff.*dl.*sin) Jm
    d/dtheta [ Mass*g * Jm' (dl.*cos) ]        = -Mass*g * Jm' diag(dl.*sin) Jm

da cui

    H = nb * Jd' diag(Kb) Jd
      - Jm' diag( (nb*g*gcoeff + Mass*g) .* dl .* sin(theta_m) ) Jm

Il primo termine e' costante (energia elastica quadratica); il secondo e'
il solo contributo dipendente dalla configurazione.

PERCHE' ABBANDONARE LE DIFFERENZE FINITE
----------------------------------------
Le differenze centrate con passo relativo 1e-6 introducono su Kfull un
errore relativo ~1e-10 (dominato dall'arrotondamento, eps/h). Poiche' lo
spettro del pencil (K, M) si allarga come ~N^4 al raffinarsi della mesh,
questo rumore viene amplificato sull'autovalore piu' soffice e ne rovina
la convergenza per N grande, mascherando l'ordine quadratico atteso.
L'Hessiano analitico e' esatto a meno dell'arrotondamento di macchina e
rimuove del tutto questo limite; e' inoltre molto piu' veloce, perche'
evita 2*nvar valutazioni del gradiente.

COMPATIBILITA'
--------------
Firma e valori di ritorno invariati rispetto alla versione a differenze
finite: buildStiffnessMatrix(theta_eq, p, relStep=None) -> (Kfull, G0, info).
Il parametro relStep e' mantenuto per retrocompatibilita' ed e' usato solo
se method='fd'.
"""

import numpy as np

from flecspy.elements.blade_potential import bladePotential


def buildStiffnessMatrix(theta_eq, p, relStep=None, method='analytic'):
    """
    Hessiano dell'energia potenziale rispetto alle variabili libere.

    Inputs:
        theta_eq : vettore equilibrio (qualsiasi forma; usato come colonna)
        p        : dict parametri (cd, Jd, cm, Jm, dl, dtheta, Kb, gcoeff,
                   nb, g, Mass)
        relStep  : passo relativo, usato solo se method='fd' (default 1e-6)
        method   : 'analytic' (default) oppure 'fd' (vecchia implementazione,
                   utile per validazione incrociata)

    Outputs:
        Kfull : (nvar, nvar) Hessiano simmetrizzato
        G0    : (nvar, 1)    gradiente all'equilibrio
        info  : dict con symmetryErrorBeforeSym, gradNormAtEq, relStep, method
    """
    theta_eq = np.asarray(theta_eq, dtype=float).reshape(-1, 1)
    nvar = theta_eq.size

    # Gradiente all'equilibrio (invariato rispetto alla versione precedente)
    _, G0 = bladePotential(theta_eq, p)
    G0 = np.asarray(G0, dtype=float).reshape(-1, 1)

    if method == 'fd':
        Kfull, info = _hessian_fd(theta_eq, p, relStep, nvar)
    else:
        Kfull, info = _hessian_analytic(theta_eq, p)

    info['gradNormAtEq'] = float(np.linalg.norm(G0))
    info['method'] = method
    return Kfull, G0, info


# ---------------------------------------------------------------------------
#  Hessiano analitico
# ---------------------------------------------------------------------------
def _hessian_analytic(theta_eq, p):
    cd = np.asarray(p['cd'], dtype=float).reshape(-1, 1)
    cm = np.asarray(p['cm'], dtype=float).reshape(-1, 1)
    Jd = np.asarray(p['Jd'], dtype=float)
    Jm = np.asarray(p['Jm'], dtype=float)

    dl = np.asarray(p['dl'], dtype=float).reshape(-1, 1)
    Kb = np.asarray(p['Kb'], dtype=float).reshape(-1, 1)
    gcoeff = np.asarray(p['gcoeff'], dtype=float).reshape(-1, 1)
    nb = float(p['nb'])
    g = float(p['g'])
    Mass = float(p['Mass'])

    # Mappe affini (cd non serve all'Hessiano: e' un termine costante di e)
    theta_m = cm + Jm @ theta_eq                     # (N,1)

    # --- termine elastico: nb * Jd' diag(Kb) Jd  (costante) ---
    wd = nb * Kb                                     # (N,1)
    Ke = Jd.T @ (wd * Jd)                            # broadcasting = diag(wd) @ Jd

    # --- termine gravitazionale: -Jm' diag(gtot .* sin(theta_m)) Jm ---
    # gtot raccoglie il peso della lama (gcoeff, gia' valutato a punto medio)
    # e quello del carico di tip, entrambi moltiplicati per dl.
    gtot = (nb * g * gcoeff + Mass * g) * dl         # (N,1)
    wg = gtot * np.sin(theta_m)                      # (N,1)
    Kg = Jm.T @ (wg * Jm)

    Kfull = Ke - Kg

    # Simmetrizzazione: l'Hessiano e' simmetrico per costruzione, l'asimmetria
    # residua e' solo ordine macchina (dipende dall'ordine delle somme nei
    # prodotti matriciali) ed e' un utile controllo di sanita'.
    asym = float(np.linalg.norm(Kfull - Kfull.T, 'fro'))
    Kfull = 0.5 * (Kfull + Kfull.T)

    info = {'symmetryErrorBeforeSym': asym, 'relStep': None}
    return Kfull, info


# ---------------------------------------------------------------------------
#  Hessiano a differenze finite (versione storica, per validazione)
# ---------------------------------------------------------------------------
def _hessian_fd(theta_eq, p, relStep, nvar):
    if relStep is None:
        relStep = 1e-6

    Kfull = np.zeros((nvar, nvar))
    for j in range(nvar):
        hj = relStep * max(1.0, abs(theta_eq[j, 0]))
        ej = np.zeros((nvar, 1))
        ej[j, 0] = 1.0
        _, Gp = bladePotential(theta_eq + hj * ej, p)
        _, Gm = bladePotential(theta_eq - hj * ej, p)
        Gp = np.asarray(Gp, dtype=float).reshape(-1, 1)
        Gm = np.asarray(Gm, dtype=float).reshape(-1, 1)
        Kfull[:, j] = ((Gp - Gm) / (2.0 * hj)).reshape(-1)

    asym = float(np.linalg.norm(Kfull - Kfull.T, 'fro'))
    Kfull = 0.5 * (Kfull + Kfull.T)

    info = {'symmetryErrorBeforeSym': asym, 'relStep': relStep}
    return Kfull, info


# ---------------------------------------------------------------------------
#  Auto-test: analitico vs differenze finite
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from flecspy.config import configFLECS
    from flecspy.core import build_params, solve_equilibrium

    for N in (16, 32, 64, 128,256):
        cfg = configFLECS()
        cfg['N'] = N
        p, theta0, _ = build_params(cfg)
        static = solve_equilibrium(p, cfg, theta0)
        th = static['theta']

        Ka, _, ia = buildStiffnessMatrix(th, p, method='analytic')
        Kf, _, if_ = buildStiffnessMatrix(th, p, method='fd')

        rel = np.linalg.norm(Ka - Kf, 'fro') / np.linalg.norm(Kf, 'fro')
        print(f"N = {N:4d} | ||K_an - K_fd|| / ||K_fd|| = {rel:.3e} | "
              f"asimmetria analitica = {ia['symmetryErrorBeforeSym']:.3e} | "
              f"asimmetria FD = {if_['symmetryErrorBeforeSym']:.3e}")
    print("\nAtteso: differenza relativa ~1e-9..1e-10 (e' l'errore delle FD, "
          "non dell'analitico);\nasimmetria analitica ~1e-16 relativa, "
          "molto minore di quella FD.")