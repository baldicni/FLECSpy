"""
buildConstraintReduction.py

Replica 1:1 di buildConstraintReduction.m

Costruisce la matrice di riduzione al prim'ordine che impone:
    GX.' * dtheta = 0
eliminando la variabile J.

CONVENZIONE INDICE J (CRITICO):
- In MATLAB main.m, J arriva da:  [~, J] = max(abs(GX));  -> J e' 1-based.
- Questa funzione Python riceve J GIA' 0-based (indice Python).
  In main.py useremo J = np.argmax(np.abs(GX)), che e' direttamente 0-based,
  e lo passeremo qui senza conversioni. NON sottrarre 1 a J qui dentro.
- Se hai un J 1-based proveniente da codice MATLAB, convertilo (J-1) PRIMA
  di chiamare questa funzione.

LOGICA (identica a MATLAB):
    keepIdx = tutti gli indici tranne J
    alpha   = -GX(keepIdx) / GX(J)          % dtheta_J = alpha.' * q
    Tred    = matrice (n, n-1):
                righe != J  -> identita' sulle coordinate ritenute
                riga  J     -> alpha.'

CONVENZIONI:
- GX trattato come colonna -> appiattito a 1-D internamente (GX(:)).
- Output:
    Tred    : (n, n-1) ndarray
    keepIdx : ndarray di indici 0-based ritenuti (lunghezza n-1)
    alpha   : (n-1,) ndarray
"""

import numpy as np


def buildConstraintReduction(GX, J):
    """
    Replica 1:1 di buildConstraintReduction.m

    Inputs:
        GX : gradiente del vincolo rispetto alle coordinate libere (n,) o (n,1)
        J  : indice 0-based della coordinata da eliminare (vedi nota sopra)

    Outputs:
        Tred    : (n, n-1) matrice di riduzione tale che dtheta = Tred * q
        keepIdx : (n-1,) indici 0-based ritenuti
        alpha   : (n-1,) coefficienti tali che dtheta_J = alpha.' * q
    """
    # GX = GX(:)
    GX = np.asarray(GX, dtype=float).reshape(-1)
    n = GX.size

    # if abs(GX(J)) < 1e-14, error
    if abs(GX[J]) < 1e-14:
        raise ValueError(
            'buildConstraintReduction: GX(J) is too small for stable elimination.'
        )

    # keepIdx = 1:n; keepIdx(J) = []   -> tutti tranne J
    keepIdx = np.delete(np.arange(n), J)

    # alpha = -GX(keepIdx) / GX(J)
    alpha = -GX[keepIdx] / GX[J]

    # Tred = zeros(n, n-1)
    Tred = np.zeros((n, n - 1))

    # Riempimento colonna per colonna sulle righe != J (identita')
    # MATLAB:
    #   col = 1;
    #   for i = 1:n
    #       if i ~= J, Tred(i,col) = 1; col = col+1; end
    #   end
    col = 0
    for i in range(n):
        if i != J:
            Tred[i, col] = 1.0
            col += 1

    # eliminated row: Tred(J,:) = alpha.'
    Tred[J, :] = alpha

    return Tred, keepIdx, alpha
