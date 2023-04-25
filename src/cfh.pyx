#!python
#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, log, sqrt, fabs

@cython.boundscheck(False)
@cython.wraparound(False)
def design_inequalities_equimolar_fast(np.ndarray[np.float_t, ndim=2] x, float M, float phiT0, \
                                       float zeta, np.ndarray[np.int_t, ndim=2] eps_index):
    # Construct the design inequalities for specified target phases assuming that all
    # target phases are equimolar and have the same number of enriched targets, M.
    cdef int N = x.shape[1]
    cdef int n = x.shape[0]
    cdef float phi0 = phiT0 / N
    cdef float rhs_in = -log(phi0 * M)
    cdef float rhs_ex = -log(phi0 * N * M / zeta)
    cdef neps = eps_index.max() + 1
    cdef np.ndarray[np.float_t, ndim=2] Xeq = np.zeros((n * N, neps), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=2] Xin = np.zeros((n * N, neps), dtype=np.float64)
    cdef int alpha, k, i, j
    cdef float krdel_ik, lhs
    ylisteq, ylistin = [], []
    for alpha in range(n):
        for k in range(N):
            for i in range(N):
                if i == k:
                    krdel_ik = 1.
                else:
                    krdel_ik = 0.
                for j in range(N):
                    if eps_index[i,j] >= 0:
                        lhs = x[alpha,i] * x[alpha,j] - 2. * krdel_ik * (x[alpha,j] - phi0) - phi0**2
                        if x[alpha,k] > 0:
                            Xeq[len(ylisteq),eps_index[i,j]] += 0.5 * lhs
                        else:
                            Xin[len(ylistin),eps_index[i,j]] += 0.5 * lhs
            if x[alpha,k] > 0:
                ylisteq.append(rhs_in)
            else:
                ylistin.append(rhs_ex)
    yeq = np.array(ylisteq)
    yin = np.array(ylistin)
    return Xeq[:len(ylisteq),:], yeq, Xin[:len(ylistin),:], yin

@cython.boundscheck(False)
@cython.wraparound(False)
def design_inequalities_fast(np.ndarray[np.float_t, ndim=2] x, np.ndarray[np.float_t, ndim=1] L, \
                             np.ndarray[np.float_t, ndim=1] phi0, float zeta, \
                             np.ndarray[np.int_t, ndim=2] eps_index):
    # Construct the design inequalities for specified target phases given arbitrary target-phase
    # compositions x.  It is assumed that the elements of x sum to 1.
    cdef int N = x.shape[1]
    cdef int n = x.shape[0]
    cdef neps = eps_index.max() + 1
    cdef np.ndarray[np.float_t, ndim=2] Xeq = np.zeros((n * N, neps), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=2] Xin = np.zeros((n * N, neps), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] c = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] card = np.zeros(n, dtype=np.float64)
    cdef int alpha, k, i, j
    cdef float krdel_ik, lhs, rhs_in, rhs_ex
    ylisteq, ylistin = [], []
    for alpha in range(n):
        c[alpha] = -1.
        for k in range(N):
            if x[alpha,k] > 0:
                card[alpha] += 1.
            c[alpha] += (x[alpha,k] - phi0[k]) / L[k] + phi0[k]
    for alpha in range(n):
        for k in range(N):
            for i in range(N):
                if i == k:
                    krdel_ik = 1.
                else:
                    krdel_ik = 0.
                for j in range(N):
                    if eps_index[i,j] >= 0:
                        lhs = (x[alpha,i] * x[alpha,j] - phi0[i] * phi0[j] \
                               - 2. * krdel_ik * (x[alpha,j] - phi0[j]))
                        if x[alpha,k] > 0:
                            Xeq[len(ylisteq),eps_index[i,j]] += 0.5 * L[k] * lhs
                        else:
                            Xin[len(ylistin),eps_index[i,j]] += 0.5 * L[k] * lhs
            if x[alpha,k] > 0:
                rhs_in = log(x[alpha,k] / phi0[k]) - L[k] * c[alpha]
                ylisteq.append(rhs_in)
            else:
                rhs_ex = log(zeta / (N * card[alpha] * phi0[k])) - L[k] * c[alpha]
                ylistin.append(rhs_ex)
    yeq = np.array(ylisteq)
    yin = np.array(ylistin)
    return Xeq[:len(ylisteq),:], yeq, Xin[:len(ylistin),:], yin

@cython.boundscheck(False)
@cython.wraparound(False)
def design_inequalities_muvar_fast(np.ndarray[np.float_t, ndim=2] phi, \
                                   np.ndarray[np.float_t, ndim=1] L, \
                                   float zeta, np.ndarray[np.int_t, ndim=2] eps_index):
    # Construct the design inequalities for the specified condensed-phase volume fractions phi assuming
    # that both the epsilon matrix and the mu vector are free parameters.
    # Elements of phi that are set to zero are understood to represent "excluded" components.
    cdef int N = phi.shape[1]
    cdef int n = phi.shape[0]
    cdef neps = eps_index.max() + 1
    cdef np.ndarray[np.float_t, ndim=2] Xeq = np.zeros((n * (N + 1), neps), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=2] Xin = np.zeros((n * (N + 1), neps), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=2] Yeq = np.zeros((n * (N + 1), N), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=2] Yin = np.zeros((n * (N + 1), N), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] card = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] phiT = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] cT = np.zeros(n, dtype=np.float64)
    cdef int alpha, i, j
    cdef float phi_excl
    ylisteq, ylistin = [], []
    for alpha in range(n):
        phiT[alpha] = phi[alpha,:].sum()
        cT[alpha] = phiT[alpha]
        for i in range(N):
            cT[alpha] -= phi[alpha,i] / L[i]
            if phi[alpha,i] > 0:
                card[alpha] += 1.
    for alpha in range(n):
        if card[alpha] < N:
            phi_excl = phiT[alpha] * zeta / ((N - card[alpha]) * card[alpha])
        else:
            phi_excl = 0.
        for i in range(N):
            for j in range(N):
                if eps_index[i,j] >= 0:
                    if phi[alpha,i] > 0:
                        Xeq[len(ylisteq),eps_index[i,j]] += L[i] * phi[alpha,j]
                    else:
                        Xin[len(ylistin),eps_index[i,j]] += L[i] * phi[alpha,j]
                else:
                    raise Exception("unknown eps_index")
            if phi[alpha,i] > 0:
                Yeq[len(ylisteq),i] = -L[i]
            else:
                Yin[len(ylistin),i] = -L[i]
            if phi[alpha,i] > 0:
                ylisteq.append(log(phi[alpha,i]) - L[i] * log(1. - phiT[alpha]) - (L[i] - 1.))
            else:
                ylistin.append(log(phi_excl) - L[i] * log(1. - phiT[alpha]) - (L[i] - 1.))
    for alpha in range(n):
        for i in range(N):
            for j in range(N):
                if eps_index[i,j] >= 0:
                    Xeq[len(ylisteq),eps_index[i,j]] += 0.5 * phi[alpha,i] * phi[alpha,j]
                else:
                    raise Exception("unknown eps_index")
        ylisteq.append(-log(1. - phiT[alpha]) - cT[alpha])
    yeq = np.array(ylisteq)
    yin = np.array(ylistin)
    return (Xeq[:len(ylisteq),:], Yeq[:len(ylisteq),:], yeq, \
            Xin[:len(ylistin),:], Yin[:len(ylistin),:], yin)
