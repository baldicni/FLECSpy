"""
buildBladeMaps.py

Replica 1:1 di buildBladeMaps.m

Costruisce le mappe affini:
    delta_th = cd + Jd * theta
    theta_m  = cm + Jm * theta

theta contiene:
    - N variabili se il tip e' libero          (constrainedTip = False)
    - N-1 variabili se l'angolo di tip e' fisso (constrainedTip = True)

CONVENZIONE INDICI (CRITICO):
MATLAB e' 1-based, Python 0-based. La traduzione degli indici espliciti e':
    MATLAB Jd(1,1)      -> Python Jd[0,0]
    MATLAB Jd(i,i)      -> Python Jd[i-1, i-1]   con i nel range MATLAB
    MATLAB Jd(i,i-1)    -> Python Jd[i-1, i-2]
    MATLAB Jd(N,nvar)   -> Python Jd[N-1, nvar-1]
    MATLAB for i=2:N-1  -> Python for i in range(2, N)      (i = 2..N-1)
    MATLAB for i=2:N    -> Python for i in range(2, N+1)    (i = 2..N)

Per ridurre il rischio di errore, MANTENGO le variabili-indice 'i' con lo
STESSO valore del loop MATLAB (cioe' 1-based) e sottraggo 1 SOLO al momento
dell'accesso all'array. Cosi' la corrispondenza riga-per-riga col MATLAB
resta letterale.

Outputs: Jd, Jm (N x nvar), cd, cm (vettori colonna N x 1 come ndarray (N,1)).
"""

import numpy as np


def buildBladeMaps(N, constrainedTip, ain, aout):
    """
    Replica 1:1 di buildBladeMaps.m

    Inputs:
        N              - numero di elementi (int)
        constrainedTip - bool
        ain            - clamp angle [rad]
        aout           - tip angle [rad] (usato solo se constrainedTip = True)

    Outputs:
        Jd : (N, nvar) ndarray
        Jm : (N, nvar) ndarray
        cd : (N, 1)    ndarray
        cm : (N, 1)    ndarray
    """
    if constrainedTip:
        nvar = N - 1

        Jd = np.zeros((N, nvar))
        Jm = np.zeros((N, nvar))
        cd = np.zeros((N, 1))
        cm = np.zeros((N, 1))

        # First element
        # Jd(1,1) = 1
        Jd[0, 0] = 1
        # Jm(1,1) = 0.5
        Jm[0, 0] = 0.5
        # cd(1) = -ain
        cd[0, 0] = -ain
        # cm(1) =  ain/2
        cm[0, 0] = ain / 2

        # Internal elements: MATLAB for i = 2:N-1  -> Python i = 2..N-1
        for i in range(2, N):  # i = 2, 3, ..., N-1
            # Jd(i,i)   =  1
            Jd[i - 1, i - 1] = 1
            # Jd(i,i-1) = -1
            Jd[i - 1, i - 2] = -1

            # Jm(i,i)   = 0.5
            Jm[i - 1, i - 1] = 0.5
            # Jm(i,i-1) = 0.5
            Jm[i - 1, i - 2] = 0.5

        # Last element: theta_N = aout fixed
        # Jd(N,nvar) = -1
        Jd[N - 1, nvar - 1] = -1
        # Jm(N,nvar) =  0.5
        Jm[N - 1, nvar - 1] = 0.5
        # cd(N) = aout
        cd[N - 1, 0] = aout
        # cm(N) = aout/2
        cm[N - 1, 0] = aout / 2

    else:
        nvar = N

        Jd = np.zeros((N, nvar))
        Jm = np.zeros((N, nvar))
        cd = np.zeros((N, 1))
        cm = np.zeros((N, 1))

        # First element
        # Jd(1,1) = 1
        Jd[0, 0] = 1
        # Jm(1,1) = 0.5
        Jm[0, 0] = 0.5
        # cd(1) = -ain
        cd[0, 0] = -ain
        # cm(1) =  ain/2
        cm[0, 0] = ain / 2

        # Remaining elements: MATLAB for i = 2:N  -> Python i = 2..N
        for i in range(2, N + 1):  # i = 2, 3, ..., N
            # Jd(i,i)   =  1
            Jd[i - 1, i - 1] = 1
            # Jd(i,i-1) = -1
            Jd[i - 1, i - 2] = -1

            # Jm(i,i)   = 0.5
            Jm[i - 1, i - 1] = 0.5
            # Jm(i,i-1) = 0.5
            Jm[i - 1, i - 2] = 0.5

    return Jd, Jm, cd, cm
