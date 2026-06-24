"""
conftest.py

Fixture e utility condivise per la suite di validazione FLECS.

Filosofia della suite (vedi anche test_flecs.py):
- Test DETERMINISTICI: valutano i moduli su un theta NOTO (theta0 o l'equilibrio
  di MATLAB caricato da golden file). Non coinvolgono l'ottimizzatore, quindi
  devono combaciare a epsilon di macchina (~1e-12 o meglio).
- Test FISICI: lasciano che l'ottimizzatore Python risolva e confrontano
  l'equilibrio con quello MATLAB. Tolleranza piu' larga (~1e-6) perche'
  SLSQP e fmincon sono solver diversi.

I golden file vivono in tests/golden/ (copiati dall'output MATLAB).
Tutti i golden sono stati generati con la config di riferimento:
  N=64, profilo '1-0.654*csi', partition='length',
  ain=pi/9, aout=-pi/9, xb=0.9702*lb (constrainedTip=True, constrainedLength=True).
"""

import os
import numpy as np
import sympy as sp
import pytest

# percorso dei golden file, relativo a questo conftest
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), 'golden')


def load_golden(name):
    """Carica un golden CSV come array NumPy (float)."""
    path = os.path.join(GOLDEN_DIR, name)
    return np.loadtxt(path, delimiter=',')


@pytest.fixture(scope='session')
def cfg():
    """Configurazione di riferimento (quella con cui sono stati generati i golden)."""
    from FLECSpy2.config import configFLECS
    return configFLECS()


@pytest.fixture(scope='session')
def blade(cfg):
    """
    Costanti di cella e mappe per la config di riferimento.
    Ritorna un dict con dtheta, dl, w, m, Kb, Jd, Jm, cd, cm e N.
    """
    from FLECSpy2.elements.cell_constants import cellConstants
    from FLECSpy2.elements.blade_maps import buildBladeMaps

    N = cfg['N']
    expr = sp.sympify(cfg['type'].replace('csi', 'psi'))
    dtheta, dl, w, m, Kb = cellConstants(
        N, cfg['wb'], cfg['lb'], cfg['rb'], cfg['hb'],
        cfg['Eb'], cfg['rhob'], expr, cfg['partition'])
    # config di riferimento: constrainedTip = True
    Jd, Jm, cd, cm = buildBladeMaps(N, True, cfg['ain'], cfg['aout'])
    return dict(N=N, dtheta=dtheta, dl=dl, w=w, m=m, Kb=Kb,
                Jd=Jd, Jm=Jm, cd=cd, cm=cm)


@pytest.fixture(scope='session')
def p(cfg, blade):
    """
    Struct parametri 'p' costruita esattamente come in main.py, per la config
    di riferimento (constrainedTip=True, constrainedLength=True).
    """
    m = blade['m']
    pp = dict(
        cd=blade['cd'], Jd=blade['Jd'], cm=blade['cm'], Jm=blade['Jm'],
        dl=blade['dl'].reshape(-1, 1),
        dtheta=blade['dtheta'].reshape(-1, 1),
        Kb=blade['Kb'].reshape(-1, 1),
        m=m.reshape(-1, 1),
        nb=cfg['nb'], g=cfg['g'], Mass=cfg['Mass'], xb=cfg['xb'],
        constrainedTip=True, constrainedLength=True,
    )
    pp['gcoeff'] = (np.flip(np.cumsum(np.flip(m))) - 0.5 * m).reshape(-1, 1)
    return pp


@pytest.fixture(scope='session')
def theta0(cfg, blade):
    """theta0 (condizione iniziale geometrica) come in main.py, 1-D."""
    th = cfg['ain'] + np.cumsum(blade['dtheta'][:blade['N']])
    th = th[:-1]   # constrainedTip
    return th.reshape(-1)


@pytest.fixture(scope='session')
def theta_eq_golden():
    """theta di equilibrio trovato da MATLAB (fmincon), 1-D."""
    return load_golden('golden_eq_theta.csv').reshape(-1)


# ---- helper di confronto ----

def relnorm(A, B):
    """Norma di Frobenius relativa ||A-B|| / ||B||."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        A = A.reshape(-1)
        B = B.reshape(-1)
    nb = np.linalg.norm(B)
    return np.linalg.norm(A - B) / (nb if nb > 0 else 1.0)
