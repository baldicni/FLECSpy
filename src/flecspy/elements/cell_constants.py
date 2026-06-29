"""
cellConstants.py

Replica 1:1 di cellConstants.m

Discretizzazione della geometria della lama e costanti di elemento.

NOTE SULLA FEDELTA':
- La parte simbolica di MATLAB (sym, int, solve, subs) e' replicata con SymPy.
- Il profilo `nprofile` arriva qui come espressione SymPy nella variabile 'psi'.
  In MATLAB la variabile simbolica e' 'psi'; qui usiamo lo stesso simbolo.
- Nel ramo 'area' MATLAB esegue, per ogni i:
      sol = solve(narea == i*dnA, psi);
      sol = sol(sol > 0);
      sol = min(sol(sol <= 1));
  Replichiamo ESATTAMENTE questa sequenza:
    1) risolvi narea == i*dnA
    2) tieni solo le soluzioni reali positive (sol > 0)
    3) tra quelle, tieni le <= 1
    4) prendi il minimo
  Le radici complesse vengono scartate prima del confronto con le soglie
  (in MATLAB 'sol > 0' su un valore complesso da' false, quindi e' equivalente).

CONVENZIONE INDICI:
- MATLAB e' 1-based. Gli array csi, da, dl, w, dtheta, K, m sono restituiti
  come array NumPy 1-D di lunghezza N (indici 0..N-1).
- csil = [0, csi(1:end-1)] -> csi spostato di una posizione con 0 davanti.
"""

import numpy as np
import sympy as sp


def cellConstants(N, w0, L, R0, h, E, rho, nprofile, part=None):
    """
    Replica 1:1 di cellConstants.m

    Inputs:
        N        - number of elements (int)
        w0       - base width
        L        - blade length
        R0       - rest curvature radius
        h        - thickness
        E        - Young modulus
        rho      - density
        nprofile - espressione SymPy del profilo normalizzato, funzione di 'psi' in [0,1]
        part     - 'length' or 'area' (default: 'length' se None)

    Outputs (tutti array NumPy 1-D di lunghezza N, dtype float):
        dtheta   - rest angle increments of each element
        dl       - element lengths
        w        - element widths
        m        - element masses
        K        - element bending spring constants
    """
    psi = sp.symbols('psi')

    # narea = int(nprofile, 0, psi)  -> area normalizzata da 0 a psi
    # Integrale indefinito valutato come integrale definito da 0 a psi.
    narea = sp.integrate(nprofile, (psi, 0, psi))

    csi = np.zeros(N, dtype=float)

    if part is not None and part == 'area':
        # Equal partition of area
        # dnA = double(subs(narea, psi, 1)) / N
        dnA = float(narea.subs(psi, 1)) / N

        for i in range(1, N + 1):  # MATLAB 1:N
            # sol = solve(narea == i*dnA, psi)
            sol = sp.solve(sp.Eq(narea, i * dnA), psi)

            # sol = sol(sol > 0): tieni solo soluzioni reali e positive.
            # Una radice complessa non soddisfa '> 0' in MATLAB -> scartata.
            real_pos = []
            for s in sol:
                s_simpl = sp.nsimplify(s) if s.free_symbols else s
                # Forziamo valutazione numerica; scartiamo i complessi.
                val = complex(s.evalf())
                if abs(val.imag) < 1e-12 and val.real > 0:
                    real_pos.append(val.real)

            # sol = min(sol(sol <= 1))
            le_one = [v for v in real_pos if v <= 1]
            if len(le_one) == 0:
                raise ValueError(
                    f"cellConstants: nessuna soluzione reale in (0,1] per i={i}."
                )
            csi[i - 1] = min(le_one)
    else:
        # Equal partition of length
        # csi = (1:N) / N
        csi = np.arange(1, N + 1, dtype=float) / N

    # csil = [0, csi(1:end-1)]
    csil = np.concatenate(([0.0], csi[:-1]))
    # csir = csi
    csir = csi

    # dl = L * (csir - csil)   -> element lengths
    dl = L * (csir - csil)

    # da = zeros(1,N); calcolo aree per elemento via narea valutata ai nodi
    da = np.zeros(N, dtype=float)
    nareal = 0.0
    for i in range(N):  # MATLAB 1:N -> 0..N-1
        # narear = double(subs(narea, psi, csir(i)))
        narear = float(narea.subs(psi, csir[i]))
        # da(i) = w0 * L * (narear - nareal)
        da[i] = w0 * L * (narear - nareal)
        nareal = narear

    # w = da ./ dl
    w = da / dl
    # dtheta = dl / R0
    dtheta = dl / R0
    # K = E * h^3 * (w ./ dl) / 12
    K = E * h**3 * (w / dl) / 12.0
    # m = rho * da * h
    m = rho * da * h

    return dtheta, dl, w, m, K
