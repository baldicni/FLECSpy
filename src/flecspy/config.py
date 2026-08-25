"""
config.py

Put here all the geometric, material and configuration characteristcs of your GAS system. 
To create a new configuration, define a new function with the desired name and fill al the fields.
To use it in a script, import it as

from flecspy.config import "configrationname"

and assign it to a variable as

cfg = configname()

"""

import numpy as np


def configFLECS():
    
    cfg = {}

    # %% Shape
    cfg['wb'] = 49e-3            # Blade base width [m]
    cfg['hb'] = 0.35e-3          # Blade thickness [m]
    cfg['lb'] = 52e-3  #52e-3    # Blade length [m]
    cfg['rb'] = 1e9              # Blade rest curvature [m]
    cfg['ain'] = np.pi / 9       # Clamp angle [rad]
    cfg['aout'] = -np.pi / 9     # Tip angle ['none' or numeric]
    cfg['type'] = '1-0.654*csi'  # Profile law
    cfg['xb'] = 0.9702 * cfg['lb']  # Horizontal projection ['none' or numeric]

    # %% Material
    cfg['Eb'] = 210e9            # Young modulus [Pa]
    cfg['UTSb'] = 1.4e9          # UTS [Pa]
    cfg['yldb'] = 1.15e9         # Yield [Pa]
    cfg['rhob'] = 7.85e3         # Density [kg/m^3]

    # %% Environment
    cfg['g'] = 9.81              # Gravity [m/s^2]
    cfg['nb'] = 1                # Number of blades
    cfg['Mass'] = 0.41         # Tip mass [kg]

    # %% Algorithm
    cfg['N'] = 64             # Number of blade partitions
    cfg['neig'] = 6             # Number of modes
    cfg['partition'] = 'length'  # 'length' or 'area'
    cfg['optimizer'] = 'SLSQP'   # 'SLSQP' (default, fast) or 'trust-constr' (slower, more accurate on theta)

    return cfg



def configCRYOGAS():

    cfg = {}

    # %% Shape
    cfg['wb'] = 33.55*0.8317*1e-3 #cfg['wb']   = 33.55e-3             # Blade base width [m]
    cfg['hb'] = 0.6e-3          # Blade thickness [m]
    cfg['lb'] = 133.64e-3 #cfg['lb']   = 103.64e-3            # Blade length [m]
    cfg['rb'] = 1e9              # Blade rest curvature [m]
    cfg['ain'] = np.pi / 4       # Clamp angle [rad]
    cfg['aout'] = -33*np.pi/180    # Tip angle ['none' or numeric]
    cfg['type'] = '(1+ 0.9328*csi - 1.9631*csi**2 + 5.8534*csi**3 - 12.22*csi**4 + 6.6757*csi**5)/0.8317' #cfg['type'] = '0.93230 + 0.47624*csi - 0.46005*csi**2 + 0.22426*csi**3 - 2.79396*csi**4 + 1.87263*csi**5' cfg['type'] = '0.8317 + 0.9328*csi - 1.9631*csi**2 + 5.8534*csi**3 - 12.22*csi**4 + 6.6757*csi**5'  # Profile law
    cfg['xb'] = 0.90*cfg['lb'] # Horizontal projection ['none' or numeric]

    # %% Material
    cfg['Eb'] = 125e9            # Young modulus [Pa]
    cfg['UTSb'] = 1.1e9          # UTS [Pa]
    cfg['yldb'] = 1.0e9         # Yield [Pa]
    cfg['rhob'] = 4.46e3         # Density [kg/m^3]

    # %% Environment
    cfg['g'] = 9.81              # Gravity [m/s^2]
    cfg['nb'] = 6                # Number of blades
    cfg['Mass'] = 8           # Tip mass [kg]

    # %% Algorithm
    cfg['N'] = 64              # Number of blade partitions
    cfg['neig'] = 1             # Number of modes
    cfg['partition'] = 'length'  # 'length' or 'area'
    cfg['optimizer'] = 'SLSQP'   # 'SLSQP' (default, fast) or 'trust-constr' (slower, more accurate on theta)

    return cfg

def configCELLA():
    """
    Configuration reproducing the theoretical blade of
      G. Cella, V. Sannibale, R. DeSalvo, S. Marka, A. Takamori,
      "Monolithic geometric anti-spring blades",
      Nucl. Instrum. Methods Phys. Res. A 540 (2005) 502-519.

    Purpose: cross-check FLECS against the dimensionless critical point of the
    paper's running example (Section 2.1.2, Figs 4-6):
        |G*_y| ~= 1.6895   (critical dimensionless vertical load)
          x*   ~= 0.9027   (critical normalized horizontal tip position X/L)

    The paper models a MASSLESS elastic line (Eq. 1) loaded by a point force at
    the tip, with zero rest curvature. This config reproduces that idealization:
      - 'tama' width profile  g^{-1}(x) = c1 + c2 cos(b x) + c3 sin(b x)
        (c1=-0.377, c2=1.377, c3=0.195, b=1.361) -> already in build_params.
      - clamp/tip angles ain = +pi/4, aout = -pi/6  (theta0, thetaL of the paper).
      - rb -> large   => negligible rest curvature (flat unstressed blade, Eq. 1).
      - rhob -> small  => negligible blade weight (massless-blade assumption).

    The dimensional scales (Eb, hb, wb, lb, g) are ARBITRARY: the paper's
    results are dimensionless and invariant under them. They only fix the
    conversion  Gy = 12 lb^2 / (Eb hb^3 wb) * (Mass * g).  Values are chosen so
    that |Gy| = 1.6895 corresponds to a tip mass of order 1 kg.
    """
    import numpy as np
    cfg = {}

    # %% Shape (Cella 2005 running example)
    cfg['wb']   = 0.05            # Blade base width w(0) [m]   (arbitrary scale)
    cfg['hb']   = 1.0e-3          # Blade thickness d [m]       (arbitrary scale)
    cfg['lb']   = 0.30           # Blade length L [m]          (arbitrary scale)
    cfg['rb']   = 1e9            # Rest curvature radius [m] -> ~flat rest (Eq. 1)
    cfg['ain']  = np.pi / 4       # theta0 = +45 deg (base/clamp angle)
    cfg['aout'] = -np.pi / 6      # thetaL = -30 deg (tip angle)
    cfg['type'] = 'tama'          # g^{-1}(x) = c1 + c2 cos(bx) + c3 sin(bx)
    cfg['xb']   = 0.9027 * cfg['lb']   # X = x* L  (critical horizontal position)

    # %% Material (arbitrary: results are dimensionless-invariant)
    cfg['Eb']   = 186e9           # Young modulus [Pa] (maraging-like, arbitrary)
    cfg['UTSb'] = 1.4e9           # UTS [Pa]   (unused here)
    cfg['yldb'] = 1.15e9          # Yield [Pa] (unused here)
    cfg['rhob'] = 0.01         # Density [kg/m^3] -> ~massless blade (Eq. 1)

    # %% Environment
    cfg['g']    = 9.81            # Gravity [m/s^2]
    cfg['nb']   = 1              # Single blade (radial symmetry -> one blade)
    cfg['Mass'] = 1.78           # Tip mass [kg] placeholder (~ |Gy|=1.6895);
    #                              overridden by the load sweep.

    # %% Algorithm
    cfg['N']         = 64        # Blade partitions (overridden in convergence study)
    cfg['neig']      = 6         # Modes (only the fundamental f1 is used)
    cfg['partition'] = 'length'   # uniform arc length -> matches x = l/L of the paper
    cfg['optimizer'] = 'SLSQP'    # near the critical point 'trust-constr' is safer

    return cfg