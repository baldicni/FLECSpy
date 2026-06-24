"""
flecs_plots.py

Funzioni di plot dedicate, separate dal calcolo. Ognuna prende i dict
'static' / 'dynamics' (e i parametri) e disegna una figura. Il chiamante
(es. main) le invoca solo se vuole vedere i grafici: questo rende superfluo
il vecchio flag plotOut.

matplotlib viene importato dentro le funzioni cosi' il core non dipende da
matplotlib se i plot non servono.
"""

import numpy as np


def plot_blade_shape(p, extra, fignum=1):
    """Forma della lama (larghezza lungo l'asse), come figure(1) di main.m."""
    import matplotlib.pyplot as plt
    dl = p['dl'].reshape(-1)
    w = extra['w']
    wb = extra['wb']
    x = np.concatenate(([0.0], np.cumsum(dl)))
    wp = np.concatenate(([wb], w))
    plt.figure(fignum); plt.clf()
    plt.plot(x, wp / 2, linewidth=2, color='black')
    plt.plot(x, -wp / 2, linewidth=2, color='black')
    plt.plot([0, 0], [-wb / 2, wb / 2], linewidth=2, color='black')
    plt.plot([x[-1], x[-1]], [-w[-1] / 2, w[-1] / 2], linewidth=2, color='black')
    plt.xlim([0, x[-1]])
    plt.xlabel('l [m]'); plt.ylabel('y [m]')
    plt.axis('equal'); plt.grid(True); plt.box(True)
    plt.title('Blade shape')
    


def plot_profile(static, fignum=2):
    """Profilo di equilibrio x-y, come figure(2) di main.m."""
    import matplotlib.pyplot as plt
    plt.figure(fignum); plt.clf()
    plt.plot(static['x'], static['y'], linewidth=2)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.axis('equal'); plt.grid(True); plt.box(True)
    plt.title('Profile')
    


def plot_stress(static, p, cfg, fignum=3):
    """Stress lungo la lama con soglie yield/UTS, come figure(3) di main.m."""
    import matplotlib.pyplot as plt
    dl = p['dl'].reshape(-1)
    s = static['stress']
    N = cfg['N']
    yldb, UTSb = cfg['yldb'], cfg['UTSb']
    ll = np.cumsum(dl)
    plt.figure(fignum); plt.clf()
    plt.plot(ll, s, linewidth=2)
    plt.plot(ll, yldb * np.ones(N), ':', linewidth=2, color='black')
    plt.plot(ll, UTSb * np.ones(N), ':', linewidth=2, color='black')
    plt.ylim([0, max(np.max(s), UTSb)])
    plt.grid(True); plt.box(True)
    plt.xlabel('l [m]'); plt.ylabel('s [Pa]')
    plt.title('Stress')
    


def plot_modes(static, dynamics, p, cfg):
    """
    Ricostruzione e plot dei modi (angoli, profilo oscillante, perturbazione
    verticale), come figure(4/5/6) di main.m. Replica la stessa normalizzazione
    L2 a 1 grado e la stessa convenzione di segno.
    """
    import matplotlib.pyplot as plt
    dl = p['dl'].reshape(-1)
    neig = cfg['neig']
    constrainedTip = p['constrainedTip']
    Tred = dynamics['Tred']
    modes = dynamics['modes']
    theta_m = static['theta_m']
    x, y = static['x'], static['y']

    ll = np.concatenate(([0.0], np.cumsum(dl)))

    fig4 = plt.figure(4); plt.clf()
    fig5 = plt.figure(5); plt.clf()
    fig6 = plt.figure(6); plt.clf()
    legth, legy, legdy = [], [], []

    for ip in range(neig):
        qmode = modes[:, ip]
        dtheta_free = Tred @ qmode
        if constrainedTip:
            dtheta_elem = np.concatenate([dtheta_free.reshape(-1), [0.0]])
        else:
            dtheta_elem = dtheta_free.reshape(-1)
        dths = np.concatenate([[0.0], dtheta_elem])

        seg = 0.5 * (dths[1:] + dths[:-1])
        L2Norm = np.sqrt(np.sum(dl * seg**2) / np.sum(dl))
        dths = (np.pi / 180) * dths / L2Norm
        if dths[1] - dths[0] < 0:
            dths = -dths

        plt.figure(4)
        plt.plot(ll, dths, linewidth=2)
        legth.append('Mode {}'.format(ip + 1))

        plt.figure(5)
        dths_m = (dths[1:] + dths[:-1]) / 2
        xs = np.concatenate(([0.0], np.cumsum(dl * np.cos(theta_m + dths_m))))
        ys = np.concatenate(([0.0], np.cumsum(dl * np.sin(theta_m + dths_m))))
        plt.plot(xs, ys, linewidth=2)
        legy.append('Mode {}'.format(ip + 1))

        plt.figure(6)
        y0s = np.interp(xs, x, y)
        plt.plot(ll, ys - y0s, linewidth=2)
        legdy.append('Mode {}'.format(ip + 1))

    plt.figure(4)
    plt.xlabel('l [m]'); plt.ylabel(r'$\theta$ [rad]')
    plt.box(True); plt.grid(True)
    plt.legend(legth, loc='lower left')
    plt.title('Eigen states - Angles')

    plt.figure(5)
    plt.plot(x, y, linewidth=2, linestyle='--', color='#808080')
    legy.append('Rest curve')
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.box(True); plt.grid(True)
    plt.legend(legy, loc='lower center')
    plt.title('Eigen states - Profile')

    plt.figure(6)
    plt.xlabel('l [m]'); plt.ylabel(r'$\delta y$ [m]')
    plt.box(True); plt.grid(True)
    plt.legend(legdy, loc='upper left')
    plt.title('Eigen states - Perturbation')
    
