"""
buildStiffnessMatrix.py

Replica 1:1 di buildStiffnessMatrix.m

Costruisce l'Hessiano numerico dell'energia potenziale tramite differenze
finite del gradiente analitico restituito da bladePotential.

PUNTI DELLA TRADUZIONE:
- relStep default 1e-6 (come MATLAB: if nargin<3 || isempty(relStep)).
- passo per colonna: hj = relStep * max(1, abs(theta_eq(j)))
- differenze centrate colonna per colonna sul GRADIENTE:
    Kfull(:,j) = (Gp - Gm) / (2*hj)
- simmetrizzazione: asym = norm(Kfull - Kfull.','fro'); Kfull = 0.5*(Kfull+Kfull.')
- bladePotential restituisce (U, G); qui usiamo solo G -> indice [1].

CONVENZIONI:
- theta_eq trattato come colonna (n,1).
- Kfull (nvar, nvar), G0 (nvar, 1), info dict.

MATLAB ".'" -> '.T'
"""

import numpy as np
from FLECSpy2.elements.blade_potential import bladePotential


def buildStiffnessMatrix(theta_eq, p, relStep=None):
    """
    Replica 1:1 di buildStiffnessMatrix.m

    Inputs:
        theta_eq : vettore equilibrio (qualsiasi forma; usato come colonna)
        p        : dict parametri (passato a bladePotential)
        relStep  : passo relativo per le differenze finite (default 1e-6)

    Outputs:
        Kfull : (nvar, nvar) Hessiano simmetrizzato
        G0    : (nvar, 1)    gradiente all'equilibrio
        info  : dict con symmetryErrorBeforeSym, gradNormAtEq, relStep
    """
    # theta_eq = theta_eq(:)
    theta_eq = np.asarray(theta_eq, dtype=float).reshape(-1, 1)
    nvar = theta_eq.size

    # if nargin < 3 || isempty(relStep), relStep = 1e-6
    if relStep is None:
        relStep = 1e-6

    # Gradient at equilibrium: [~, G0] = bladePotential(theta_eq, p); G0 = G0(:)
    _, G0 = bladePotential(theta_eq, p)
    G0 = np.asarray(G0, dtype=float).reshape(-1, 1)

    Kfull = np.zeros((nvar, nvar))

    # Central finite differences column by column
    for j in range(nvar):  # MATLAB 1:nvar
        # hj = relStep * max(1, abs(theta_eq(j)))
        hj = relStep * max(1.0, abs(theta_eq[j, 0]))
        # ej = zeros(nvar,1); ej(j) = 1
        ej = np.zeros((nvar, 1))
        ej[j, 0] = 1.0

        # [~, Gp] = bladePotential(theta_eq + hj*ej, p)
        _, Gp = bladePotential(theta_eq + hj * ej, p)
        # [~, Gm] = bladePotential(theta_eq - hj*ej, p)
        _, Gm = bladePotential(theta_eq - hj * ej, p)

        Gp = np.asarray(Gp, dtype=float).reshape(-1, 1)
        Gm = np.asarray(Gm, dtype=float).reshape(-1, 1)

        # Kfull(:,j) = (Gp - Gm) / (2*hj)
        Kfull[:, j] = ((Gp - Gm) / (2.0 * hj)).reshape(-1)

    # Symmetrize to remove numerical asymmetry
    # asym = norm(Kfull - Kfull.', 'fro')
    asym = np.linalg.norm(Kfull - Kfull.T, 'fro')
    # Kfull = 0.5 * (Kfull + Kfull.')
    Kfull = 0.5 * (Kfull + Kfull.T)

    info = {
        'symmetryErrorBeforeSym': asym,
        'gradNormAtEq': np.linalg.norm(G0),
        'relStep': relStep,
    }

    return Kfull, G0, info
