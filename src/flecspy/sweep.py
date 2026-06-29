"""
flecs_sweep.py

Primitive di sweep parametrico per FLECS, condivise dai vari script di analisi.

Tutta la fisica di sweep vive qui; i singoli script (sweep_freq_vs_load,
sweep_freq_2d, ...) la usano e si occupano solo del plot specifico.

Idee chiave:
  - Si costruisce 'p' UNA volta con build_params; le mappe (Jd/Jm/cd/cm) non
    dipendono da Mass ne' da xb, quindi non vanno ricostruite a ogni punto.
  - Warm start: l'equilibrio trovato a un punto e' il punto di partenza
    (theta0) per il punto successivo -> convergenza piu' rapida e robusta,
    perche' punti vicini hanno equilibri vicini.
  - Le funzioni restituiscono dati grezzi (array). Il plotting e il salvataggio
    sono responsabilita' del chiamante.

Convenzione: 'compression' indica xb/lb (frazione adimensionale della
lunghezza), perche' e' la quantita' fisicamente parlante. xb = compression*lb.
"""

import copy
import numpy as np

from flecspy.core import build_params, solve_equilibrium, solve_dynamics


def _clone_params(p):
    """Copia profonda di p, per non sporcare l'originale variando Mass/xb."""
    return copy.deepcopy(p)


def sweep_load_1d(cfg, mass_vec, compression=None, want_dynamics=True,
                  warm_start=True, verbose=False):
    """
    Sweep 1D sul carico verticale (Mass), a compressione fissa.

    Inputs:
        cfg         : config FLECS
        mass_vec    : array dei valori di Mass [kg] da esplorare
        compression : xb/lb da usare (float). Se None, usa cfg['xb']/cfg['lb'].
        want_dynamics : se True calcola anche f1 e le frequenze (piu' lento);
                        se False solo statica (tip, lambda, stress).
        warm_start  : se True usa l'equilibrio precedente come theta0 successivo.
        verbose     : se True stampa una riga per punto.

    Returns: dict con array allineati a mass_vec:
        mass, f1 (o None), freqs (lista, o None), lambda_, htip, xtip,
        smax, tip_angle, converged (bool per punto)
    """
    p, theta0_base, extra = build_params(cfg)
    lb = extra['lb']

    if compression is not None:
        xb = compression * lb
    else:
        xb = cfg['xb']

    p = _clone_params(p)
    p['xb'] = xb
    p['constrainedLength'] = True  # sweep di carico a lunghezza vincolata

    n = len(mass_vec)
    f1 = np.full(n, np.nan)
    lam = np.full(n, np.nan)
    htip = np.full(n, np.nan)
    xtip = np.full(n, np.nan)
    smax = np.full(n, np.nan)
    tipang = np.full(n, np.nan)
    converged = np.zeros(n, dtype=bool)
    freqs_all = [None] * n

    theta_warm = theta0_base
    for k, M in enumerate(mass_vec):
        p['Mass'] = float(M)
        try:
            static = solve_equilibrium(p, cfg, theta_warm if warm_start else theta0_base)
            lam[k] = static['lambda_']
            htip[k] = static['htip']
            xtip[k] = static['xtip']
            smax[k] = static['smax']
            tipang[k] = static['tipAngle']
            converged[k] = True

            if want_dynamics:
                dyn = solve_dynamics(p, static, cfg)
                freqs_all[k] = dyn['freqs']
                f1[k] = dyn['freqs'][0]

            if warm_start:
                theta_warm = static['theta']
        except Exception as e:
            if verbose:
                print(f'  [Mass={M:.4g}] not convergent: {e}')

        if verbose:
            if want_dynamics:
                print(f'  Mass={M:.4g} kg  f1={f1[k]:.6g} Hz  '
                  f'lambda={lam[k]:.4g} N  htip={htip[k]*1e3:.4g} mm  '
                  f'smax={smax[k]/1e6:.4g} MPa')
            else:
                print(f'  Mass={M:.4g} kg'
                  f'lambda={lam[k]:.4g} N  htip={htip[k]*1e3:.4g} mm  '
                  f'smax={smax[k]/1e6:.4g} MPa')

    return dict(mass=np.asarray(mass_vec, dtype=float),
                f1=f1, freqs=freqs_all, lambda_=lam,
                htip=htip, xtip=xtip, smax=smax, tip_angle=tipang,
                converged=converged, compression=(xb / lb))


def sweep_load_compression_2d(cfg, mass_vec, comp_vec, want_dynamics=True,
                              warm_start=True, verbose=False):
    """
    Sweep 2D su (Mass, compression). Per ogni compressione fa uno sweep 1D in
    Mass, riusando il warm start sia lungo Mass sia tra righe di compressione.

    Inputs:
        cfg       : config
        mass_vec  : array Mass [kg]      (asse colonne)
        comp_vec  : array xb/lb          (asse righe)
        want_dynamics : se True calcola f1 (necessario per heatmap di frequenza)

    Returns: dict con griglie di forma (len(comp_vec), len(mass_vec)):
        mass_vec, comp_vec, F1, LAMBDA, HTIP, SMAX, CONVERGED
        (righe = compression, colonne = Mass; come imagesc/meshgrid)
    """
    nC = len(comp_vec)
    nM = len(mass_vec)
    F1 = np.full((nC, nM), np.nan)
    LAM = np.full((nC, nM), np.nan)
    HTIP = np.full((nC, nM), np.nan)
    SMAX = np.full((nC, nM), np.nan)
    CONV = np.zeros((nC, nM), dtype=bool)

    # warm start tra righe: l'equilibrio della riga precedente (a parita' di
    # Mass iniziale) e' un buon punto di partenza per la riga successiva
    p, theta0_base, extra = build_params(cfg)
    theta_row_seed = theta0_base

    for ic, comp in enumerate(comp_vec):
        # sweep 1D su questa riga; usiamo il seed dalla riga precedente come
        # punto di partenza del primo Mass
        res = _sweep_row(cfg, mass_vec, comp, want_dynamics, warm_start,
                         row_seed=theta_row_seed, extra=extra,
                         theta0_base=theta0_base)
        F1[ic, :] = res['f1']
        LAM[ic, :] = res['lambda_']
        HTIP[ic, :] = res['htip']
        SMAX[ic, :] = res['smax']
        CONV[ic, :] = res['converged']
        # aggiorna il seed per la prossima riga col primo equilibrio convergente
        if res['first_theta'] is not None:
            theta_row_seed = res['first_theta']

        if verbose:
            ok = int(np.sum(res['converged']))
            print(f'compression={comp:.4f}  (row {ic + 1}/{nC})  '
                  f'convergent: {ok}/{nM}')

    return dict(mass_vec=np.asarray(mass_vec, dtype=float),
                comp_vec=np.asarray(comp_vec, dtype=float),
                F1=F1, LAMBDA=LAM, HTIP=HTIP, SMAX=SMAX, CONVERGED=CONV)


def _sweep_row(cfg, mass_vec, compression, want_dynamics, warm_start,
               row_seed, extra, theta0_base):
    """Helper: uno sweep 1D in Mass per una compressione, con seed esterno."""
    lb = extra['lb']
    p, _, _ = build_params(cfg)
    p = _clone_params(p)
    p['xb'] = compression * lb
    p['constrainedLength'] = True

    n = len(mass_vec)
    f1 = np.full(n, np.nan)
    lam = np.full(n, np.nan)
    htip = np.full(n, np.nan)
    smax = np.full(n, np.nan)
    conv = np.zeros(n, dtype=bool)
    first_theta = None

    theta_warm = row_seed if row_seed is not None else theta0_base
    for k, M in enumerate(mass_vec):
        p['Mass'] = float(M)
        try:
            static = solve_equilibrium(p, cfg, theta_warm if warm_start else theta0_base)
            lam[k] = static['lambda_']
            htip[k] = static['htip']
            smax[k] = static['smax']
            conv[k] = True
            if want_dynamics:
                dyn = solve_dynamics(p, static, cfg)
                f1[k] = dyn['freqs'][0]
            if warm_start:
                theta_warm = static['theta']
            if first_theta is None:
                first_theta = static['theta']
        except Exception:
            pass

    return dict(f1=f1, lambda_=lam, htip=htip, smax=smax,
                converged=conv, first_theta=first_theta)


def save_csv(path, columns, header):
    """
    Salva colonne in un CSV. 'columns' lista di array 1-D della stessa lunghezza,
    'header' lista di nomi. Nessun percorso hardcoded: il chiamante decide dove.
    """
    data = np.column_stack(columns)
    np.savetxt(path, data, delimiter=',', header=','.join(header),
               comments='', fmt='%.10g')
    
def save_matrix_csv(path, row_vals, col_vals, M, corner='', fmt='%.10g'):
    """
    Salva una matrice 2-D in CSV, formato 'wide':
      - prima riga:  corner, col_vals[0], col_vals[1], ...
      - riga i:      row_vals[i], M[i,0], M[i,1], ...
    'M' ha forma (len(row_vals), len(col_vals)). I NaN sono scritti come 'nan'.
    Pensata per griglie 2-D (es. f1 in funzione di compressione x Mass).
    """
    M = np.asarray(M, dtype=float)
    nr, nc = M.shape
    if len(row_vals) != nr or len(col_vals) != nc:
        raise ValueError(
            f'Incoherent dimensions: M is {nr}x{nc}, '
            f'row_vals={len(row_vals)}, col_vals={len(col_vals)}')
    lines = [','.join([str(corner)] + [fmt % c for c in col_vals])]
    for i in range(nr):
        lines.append(','.join([fmt % row_vals[i]] + [fmt % v for v in M[i]]))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')