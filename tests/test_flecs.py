"""
test_flecs.py

Suite di validazione della traduzione Python di FLECS contro golden file MATLAB.

Esecuzione:  pytest -v
             pytest -v -m physical      (solo test fisici, equilibrio)
             pytest -v -m "not physical" (solo deterministici, veloci)

Le tolleranze sono state determinate empiricamente durante il porting e
documentate inline. Due regimi:
  - DETERMINISTICO (moduli su theta noto): ~1e-12 o meglio.
  - FISICO (equilibrio da ottimizzatore): ~1e-6 (SLSQP vs fmincon).

Nota sulle tolleranze speciali:
  - bladeClampMoments.M_in: cancellazione catastrofica (somma di termini di
    segno opposto e modulo simile) -> tolleranza relativa ~1e-10.
  - freqs: dipende dal solver agli autovalori (eigh vs eigs) e dal modo
    fondamentale 'morbido' (anti-spring) mal condizionato -> ~1e-6 sul Tipo B.
"""

import numpy as np
import sympy as sp
import pytest

from conftest import load_golden, relnorm


# =====================================================================
#  cellConstants  (DETERMINISTICO, ramo 'length')
# =====================================================================
class TestCellConstants:
    def test_against_golden(self, cfg):
        from FLECSpy2.elements.cell_constants import cellConstants
        N = cfg['N']
        expr = sp.sympify(cfg['type'].replace('csi', 'psi'))
        dtheta, dl, w, m, K = cellConstants(
            N, cfg['wb'], cfg['lb'], cfg['rb'], cfg['hb'],
            cfg['Eb'], cfg['rhob'], expr, cfg['partition'])
        py = np.column_stack([dtheta, dl, w, m, K])
        golden = load_golden('golden_cellConstants.csv')
        assert py.shape == golden.shape
        # norma relativa colonna per colonna (la simbolica SymPy vs MATLAB
        # arrotonda diversamente -> ~1e-14)
        assert np.allclose(py, golden, rtol=1e-12, atol=0)

    def test_length_sums_to_lb(self, cfg, blade):
        # check fisico: la somma delle lunghezze elemento = lunghezza lama
        assert np.isclose(blade['dl'].sum(), cfg['lb'], rtol=1e-12)


# =====================================================================
#  buildBladeMaps  (DETERMINISTICO, entrambi i branch)
# =====================================================================
class TestBladeMaps:
    @pytest.mark.parametrize('branch,suffix', [(True, 'tipTrue'), (False, 'tipFalse')])
    def test_against_golden(self, cfg, branch, suffix):
        from FLECSpy2.elements.blade_maps import buildBladeMaps
        Jd, Jm, cd, cm = buildBladeMaps(cfg['N'], branch, cfg['ain'], cfg['aout'])
        # Jd, Jm sono di soli 0/+-1/0.5 -> attesi bit-per-bit
        assert np.allclose(Jd, load_golden(f'golden_Jd_{suffix}.csv'), rtol=0, atol=1e-14)
        assert np.allclose(Jm, load_golden(f'golden_Jm_{suffix}.csv'), rtol=0, atol=1e-14)
        # cd, cm contengono ain/aout -> epsilon di macchina sull'arrotondamento
        assert np.allclose(cd.reshape(-1), load_golden(f'golden_cd_{suffix}.csv').reshape(-1), rtol=1e-12, atol=1e-15)
        assert np.allclose(cm.reshape(-1), load_golden(f'golden_cm_{suffix}.csv').reshape(-1), rtol=1e-12, atol=1e-15)


# =====================================================================
#  bladePotential  (DETERMINISTICO su theta0)
# =====================================================================
class TestBladePotential:
    def test_U_and_G_against_golden(self, p, theta0):
        from FLECSpy2.elements.blade_potential import bladePotential
        # NB: theta0 golden e quello ricostruito coincidono a 1e-16; uso il golden
        th = load_golden('golden_theta0.csv').reshape(-1)
        U, G = bladePotential(th, p)
        U_g = float(load_golden('golden_U.csv'))
        G_g = load_golden('golden_G.csv').reshape(-1)
        assert abs(U - U_g) / abs(U_g) < 1e-12
        # gradiente: la norma relativa e' la metrica giusta (componenti vicine
        # a zero rendono il rel-per-componente fuorviante)
        assert relnorm(G.reshape(-1), G_g) < 1e-11

    def test_gradient_matches_finite_difference(self, p, theta0):
        # coerenza interna: grad analitico vs FD del potenziale
        from FLECSpy2.elements.blade_potential import bladePotential
        def U(th): return bladePotential(th, p)[0]
        _, G = bladePotential(theta0, p)
        G = G.reshape(-1)
        n = theta0.size
        Gfd = np.zeros(n)
        for j in range(n):
            h = 1e-7 * max(1, abs(theta0[j]))
            ep = theta0.copy(); ep[j] += h
            em = theta0.copy(); em[j] -= h
            Gfd[j] = (U(ep) - U(em)) / (2 * h)
        assert relnorm(G, Gfd) < 1e-7


# =====================================================================
#  buildMassMatrix  (DETERMINISTICO su theta0)
# =====================================================================
class TestMassMatrix:
    def test_Mtot_against_golden(self, p):
        from FLECSpy2.elements.build_mass_matrix import buildMassMatrix
        th = load_golden('golden_theta0.csv').reshape(-1)
        _, _, _, Mtot, _, _ = buildMassMatrix(th, p)
        assert np.allclose(Mtot, load_golden('golden_Mtot.csv'), rtol=1e-12, atol=0)

    def test_Mtot_symmetric(self, p):
        from FLECSpy2.elements.build_mass_matrix import buildMassMatrix
        th = load_golden('golden_theta0.csv').reshape(-1)
        _, _, _, Mtot, _, _ = buildMassMatrix(th, p)
        assert np.linalg.norm(Mtot - Mtot.T) < 1e-18

    def test_Mtot_positive_semidefinite(self, p):
        from FLECSpy2.elements.build_mass_matrix import buildMassMatrix
        th = load_golden('golden_theta0.csv').reshape(-1)
        _, _, _, Mtot, _, _ = buildMassMatrix(th, p)
        ev = np.linalg.eigvalsh(0.5 * (Mtot + Mtot.T))
        assert ev.min() > -1e-12   # mass matrix: definita/semidef positiva


# =====================================================================
#  buildStiffnessMatrix  (DETERMINISTICO su theta0)
# =====================================================================
class TestStiffnessMatrix:
    def test_Kfull_against_golden(self, p):
        from FLECSpy2.elements.build_stiffness_matrix import buildStiffnessMatrix
        th = load_golden('golden_theta0.csv').reshape(-1)
        Kfull, G0, info = buildStiffnessMatrix(th, p)
        # FD dell'Hessiano: stesso schema e relStep in Py e MATLAB -> ~1e-12 rel
        assert relnorm(Kfull, load_golden('golden_Kfull.csv')) < 1e-10

    def test_Kfull_symmetric(self, p):
        from FLECSpy2.elements.build_stiffness_matrix import buildStiffnessMatrix
        th = load_golden('golden_theta0.csv').reshape(-1)
        Kfull, _, _ = buildStiffnessMatrix(th, p)
        assert np.linalg.norm(Kfull - Kfull.T) < 1e-18  # simmetrizzata esplicitamente


# =====================================================================
#  buildConstraintDerivatives / bladeConstraint  (DETERMINISTICO su theta0)
# =====================================================================
class TestConstraint:
    def test_GX_HX_Xeq_against_golden(self, p):
        from FLECSpy2.elements.build_constraint_derivatives import buildConstraintDerivatives
        th = load_golden('golden_theta0.csv').reshape(-1)
        GX, HX, Xeq, _ = buildConstraintDerivatives(th, p)
        assert np.allclose(GX.reshape(-1), load_golden('golden_GX.csv').reshape(-1), rtol=1e-12, atol=1e-18)
        assert np.allclose(HX, load_golden('golden_HX.csv'), rtol=1e-12, atol=1e-18)
        assert abs(Xeq - float(load_golden('golden_Xeq.csv'))) < 1e-15

    def test_bladeConstraint_against_golden(self, p):
        from FLECSpy2.elements.blade_constraint  import bladeConstraint
        th = load_golden('golden_theta0.csv').reshape(-1)
        c, ceq, GC, GCeq = bladeConstraint(th, p)
        assert abs(ceq - float(load_golden('golden_ceq.csv'))) < 1e-12
        assert np.allclose(GCeq.reshape(-1), load_golden('golden_GCeq.csv').reshape(-1), rtol=1e-12, atol=1e-18)
        assert c.size == 0 and GC.size == 0

    def test_GX_equals_GCeq(self, p):
        # GX (da buildConstraintDerivatives) e GCeq (da bladeConstraint) sono
        # la stessa quantita': devono coincidere esattamente.
        from FLECSpy2.elements.build_constraint_derivatives  import buildConstraintDerivatives
        from FLECSpy2.elements.blade_constraint  import bladeConstraint
        th = load_golden('golden_theta0.csv').reshape(-1)
        GX, _, _, _ = buildConstraintDerivatives(th, p)
        _, _, _, GCeq = bladeConstraint(th, p)
        assert np.linalg.norm(GX.reshape(-1) - GCeq.reshape(-1)) == 0.0


# =====================================================================
#  buildConstraintReduction  (DETERMINISTICO)
# =====================================================================
class TestConstraintReduction:
    def test_against_golden(self, p):
        from FLECSpy2.elements.build_constraint_derivatives  import buildConstraintDerivatives
        from FLECSpy2.elements.build_constraint_reduction import buildConstraintReduction
        th = load_golden('golden_theta0.csv').reshape(-1)
        GX, _, _, _ = buildConstraintDerivatives(th, p)
        J = int(np.argmax(np.abs(np.asarray(GX).reshape(-1))))
        Tred, keepIdx, alpha = buildConstraintReduction(GX, J)
        assert np.allclose(Tred, load_golden('golden_Tred.csv'), rtol=1e-12, atol=1e-14)
        assert np.allclose(alpha.reshape(-1), load_golden('golden_alpha.csv').reshape(-1), rtol=1e-12, atol=1e-14)

    def test_index_convention(self, p):
        # J Python (0-based) deve essere J MATLAB (1-based) - 1
        from FLECSpy2.elements.build_constraint_derivatives  import buildConstraintDerivatives
        th = load_golden('golden_theta0.csv').reshape(-1)
        GX, _, _, _ = buildConstraintDerivatives(th, p)
        J_py = int(np.argmax(np.abs(np.asarray(GX).reshape(-1))))
        J_matlab = int(load_golden('golden_J.csv'))
        assert J_matlab == J_py + 1

    def test_reduction_satisfies_constraint(self, p):
        # proprieta' fondamentale: GX^T Tred = 0
        from FLECSpy2.elements.build_constraint_derivatives  import buildConstraintDerivatives
        from FLECSpy2.elements.build_constraint_reduction import buildConstraintReduction
        th = load_golden('golden_theta0.csv').reshape(-1)
        GX, _, _, _ = buildConstraintDerivatives(th, p)
        J = int(np.argmax(np.abs(np.asarray(GX).reshape(-1))))
        Tred, _, _ = buildConstraintReduction(GX, J)
        assert np.linalg.norm(np.asarray(GX).reshape(-1) @ Tred) < 1e-15


# =====================================================================
#  bladeClampMoments  (DETERMINISTICO su theta0, lambda fissato)
# =====================================================================
class TestClampMoments:
    def test_against_golden_lambda_minus5(self, p):
        from FLECSpy2.elements.blade_clamp_moments import bladeClampMoments
        th = load_golden('golden_theta0.csv').reshape(-1)
        M_in, M_out, dU_dain, dU_daout = bladeClampMoments(th, -5.0, p)
        g = load_golden('golden_clampMoments.csv').reshape(-1)  # [M_in,M_out,dU_dain,dU_daout]
        # M_out, dU_daout: scala ~11, nessuna cancellazione -> ~1e-13
        assert abs(M_out - g[1]) / abs(g[1]) < 1e-12
        assert abs(dU_daout - g[3]) / abs(g[3]) < 1e-12
        # M_in, dU_dain: cancellazione catastrofica -> tolleranza ~1e-10
        assert abs(M_in - g[0]) / abs(g[0]) < 1e-10
        assert abs(dU_dain - g[2]) / abs(g[2]) < 1e-10

    def test_guardrails(self, p):
        from FLECSpy2.elements.blade_clamp_moments import bladeClampMoments
        th = load_golden('golden_theta0.csv').reshape(-1)
        p2 = dict(p); p2['constrainedTip'] = False
        with pytest.raises(ValueError):
            bladeClampMoments(th, -5.0, p2)
        p3 = dict(p); p3['constrainedLength'] = False
        with pytest.raises(ValueError):
            bladeClampMoments(th, -5.0, p3)


# =====================================================================
#  TIPO A: catena a valle valutata sull'equilibrio MATLAB (DETERMINISTICO)
#  Questo e' il test FORTE: isola i moduli dall'ottimizzatore.
# =====================================================================
class TestDownstreamOnMatlabEquilibrium:
    def _downstream(self, p, cfg, theta):
        from FLECSpy2.elements.build_mass_matrix import buildMassMatrix
        from FLECSpy2.elements.build_stiffness_matrix import buildStiffnessMatrix
        from FLECSpy2.elements.build_constraint_derivatives  import buildConstraintDerivatives
        from FLECSpy2.elements.build_constraint_reduction import buildConstraintReduction
        from FLECSpy2.elements.blade_clamp_moments import bladeClampMoments
        from scipy.linalg import eigh
        _, _, _, Mtot, _, _ = buildMassMatrix(theta, p)
        Kfull, G0, _ = buildStiffnessMatrix(theta, p)
        GX, HX, _, _ = buildConstraintDerivatives(theta, p)
        GXc = np.asarray(GX).reshape(-1, 1); G0c = np.asarray(G0).reshape(-1, 1)
        lam = float((-(GXc.T @ G0c) / (GXc.T @ GXc)).item())
        Ktot = Kfull + lam * HX
        J = int(np.argmax(np.abs(np.asarray(GX).reshape(-1))))
        Tred, _, _ = buildConstraintReduction(GX, J)
        Mred = Tred.T @ Mtot @ Tred
        Kred = Tred.T @ Ktot @ Tred
        ev, _ = eigh(Kred, Mred)
        sel = np.argsort(np.abs(ev))[:cfg['neig']]
        freqs = np.sqrt(np.maximum(np.real(ev[sel]), 0)) / (2 * np.pi)
        M_in, M_out, _, _ = bladeClampMoments(theta, lam, p)
        return dict(Mtot=Mtot, Kfull=Kfull, Ktot=Ktot, GX=GX, HX=HX,
                    Mred=Mred, Kred=Kred, lam=lam, freqs=freqs,
                    M_in=M_in, M_out=M_out, J=J)

    def test_matrices(self, p, cfg, theta_eq_golden):
        d = self._downstream(p, cfg, theta_eq_golden)
        assert relnorm(d['Mtot'], load_golden('golden_eq_Mtot.csv')) < 1e-12
        assert relnorm(d['Kfull'], load_golden('golden_eq_Kfull.csv')) < 1e-10
        assert relnorm(d['Ktot'], load_golden('golden_eq_Ktot.csv')) < 1e-10
        assert relnorm(d['Mred'], load_golden('golden_eq_Mred.csv')) < 1e-12
        assert relnorm(d['Kred'], load_golden('golden_eq_Kred.csv')) < 1e-10

    def test_lambda(self, p, cfg, theta_eq_golden):
        d = self._downstream(p, cfg, theta_eq_golden)
        lam_g = float(load_golden('golden_eq_lambda.csv'))
        assert abs(d['lam'] - lam_g) / abs(lam_g) < 1e-10

    def test_freqs(self, p, cfg, theta_eq_golden):
        d = self._downstream(p, cfg, theta_eq_golden)
        fr_g = load_golden('golden_eq_freqs.csv').reshape(-1)
        # sullo stesso theta, la differenza e' solo eigh vs eigs -> ~1e-9
        assert relnorm(d['freqs'], fr_g) < 1e-8

    def test_clamp_moments(self, p, cfg, theta_eq_golden):
        d = self._downstream(p, cfg, theta_eq_golden)
        cm_g = load_golden('golden_eq_clampMoments.csv').reshape(-1)
        assert abs(d['M_in'] - cm_g[0]) / abs(cm_g[0]) < 1e-10
        assert abs(d['M_out'] - cm_g[1]) / abs(cm_g[1]) < 1e-10


# =====================================================================
#  TIPO B: equilibrio Python (SLSQP) vs MATLAB (fmincon)  [FISICO]
#  Tolleranza larga: solver diversi convergono a punti vicini ma non identici.
# =====================================================================
@pytest.mark.physical
class TestFullSolveAgainstMatlab:
    @pytest.fixture(scope='class')
    def result(self):
        # Chiamiamo direttamente le funzioni pure del core invece di main():
        # testa la stessa identica fisica senza passare per stampe ne' plot.
        from FLECSpy2.config import configFLECS
        from FLECSpy2.core import build_params, solve_equilibrium, solve_dynamics
        cfg = configFLECS()
        p, theta0, extra = build_params(cfg)
        static = solve_equilibrium(p, cfg, theta0)
        dynamics = solve_dynamics(p, static, cfg)
        # dict 'piatto' con le chiavi che i test leggono
        return dict(
            theta=static['theta'], U0=static['U0'], lambda_=static['lambda_'],
            freqs=dynamics['freqs'], M_in=static['M_in'], M_out=static['M_out'],
        )

    def test_U0(self, result):
        U0_g = float(load_golden('golden_eq_U0.csv'))
        # U0 al minimo: piattezza quadratica -> molto piu' preciso di theta
        assert abs(result['U0'] - U0_g) / abs(U0_g) < 1e-10

    def test_theta(self, result, theta_eq_golden):
        # equilibrio: SLSQP vs fmincon -> ~1e-7
        assert relnorm(result['theta'], theta_eq_golden) < 1e-5

    def test_lambda(self, result):
        lam_g = float(load_golden('golden_eq_lambda.csv'))
        assert abs(result['lambda_'] - lam_g) / abs(lam_g) < 1e-5

    def test_freqs(self, result):
        fr_g = load_golden('golden_eq_freqs.csv').reshape(-1)
        # f1 (modo morbido anti-spring) e' la piu' sensibile -> ~1e-6
        assert relnorm(result['freqs'], fr_g) < 1e-5
        assert abs(result['freqs'][0] - fr_g[0]) / abs(fr_g[0]) < 1e-5

    def test_clamp_moments(self, result):
        cm_g = load_golden('golden_eq_clampMoments.csv').reshape(-1)
        assert abs(result['M_in'] - cm_g[0]) / abs(cm_g[0]) < 1e-3
        assert abs(result['M_out'] - cm_g[1]) / abs(cm_g[1]) < 1e-5
