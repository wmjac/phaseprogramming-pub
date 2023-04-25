import math, random
import numpy as np

from fh import find_sp, eig_symm_safe

### This script provides routines for testing solutions to the FH equations.

class SolverException(Exception):
    pass

def symm_dist_matrix(c):
    dist = np.zeros((c.shape[0], c.shape[0]))
    for p in range(c.shape[0]):
        for q in range(c.shape[0]):
            dist[p,q] = np.linalg.norm(c[p,:] / c[p,:].sum() - c[q,:] / c[q,:].sum())
    return dist

def test_landscape_dilute(F, mu, phi, randoffset=0., verbose=1):
    # Test whether the dilute phase is stable by minimizing the grand potential at the prescribed
    # chemical potentials mu.
    if randoffset:
        phi *= 1 + np.array([random.gauss(0, randoffset) for i in range(len(phi))])
    try:
        phisp, omegasp, dmusp, isconv = find_sp(F, mu, phi)
    except np.linalg.LinAlgError:
        raise SolverException
    dxsp = phisp / phisp.sum() - phi / phi.sum()
    d2sp = F[2](phisp)
    eigsp, eigvsp = eig_symm_safe(d2sp)
    mineigsp = min(eigsp)
    mineigvsp = eigvsp[:,np.argmin(eigsp)]
    if verbose:
        with np.printoptions(formatter={'float': '{: 6.3f}'.format}, linewidth=240):
            print("> dil: Omega=%8.5f log10|dmu|=%6.2f dist=%5.3f phiT=%5.3f mineig=%7.3f "
                  "conv?%s" % \
                  (omegasp, math.log10(max(1.e-20, np.linalg.norm(dmusp))), \
                   np.linalg.norm(dxsp), phi.sum(), mineigsp, isconv))
    if not isconv: # Stationary point invalid:
        raise SolverException
    success = phisp.sum() < 0.5
    if verbose:
        print("LANDSCAPE DILUTE:: Success?", success)
    return {'phisp' : phisp,
            'dxsp' : dxsp,
            'omegasp' : omegasp,
            'dmusp' : dmusp,
            'eigsp' : eigsp,
            'eigvsp' : eigvsp,
            'success' : success}

def test_landscape_targets(F, mu, phi, randoffset=0., maxdist_success=0.1, verbose=1):
    # Test whether each of the target phases is stable by minimizing the grand potential
    # at the prescribed chemical potentials mu.
    phisp = np.zeros(phi.shape)
    xsp = np.zeros(phi.shape)
    dxsp = np.zeros(phi.shape)
    dmusp = np.zeros(phi.shape)
    omegasp = np.zeros(phi.shape[0])
    mineigsp = np.zeros(phi.shape[0])
    mineigvsp = np.zeros(phi.shape)
    for p in range(phi.shape[0]):
        if verbose > 1:
            with np.printoptions(formatter={'float': '{: 6.3f}'.format}, linewidth=240):
                print(">%3d: targ =" % p, phi[p,:], "phiT=%5.3f" % phi[p,:].sum())
        phiinitp = phi[p,:]
        if randoffset:
            phiinitp *= 1 + np.array([random.gauss(0, randoffset) for i in range(phi.shape[1])])
        try:
            phisp_p, omega, dmu, isconv = find_sp(F, mu, phiinitp)
        except np.linalg.LinAlgError:
            raise SolverException
        phisp[p,:] = phisp_p
        xsp[p,:] = phisp[p,:] / phisp[p,:].sum()
        dxsp[p,:] = (phisp[p,:] / phisp[p,:].sum() - phi[p,:] / phi[p,:].sum())
        dmusp[p,:] = dmu
        omegasp[p] = omega
        d2sp = F[2](phisp[p,:])
        eigsp, eigvsp = eig_symm_safe(d2sp)
        mineigsp[p] = min(eigsp)
        mineigvsp[p,:] = eigvsp[:,np.argmin(eigsp)]
        align = math.fabs(np.dot(eigvsp[:,np.argmin(eigsp)], phisp[p,:]) / np.linalg.norm(phisp[p,:]))
        if verbose:
            with np.printoptions(formatter={'float': '{: 6.3f}'.format}, linewidth=240):
                if verbose == 1:
                    print("> %3d: phiT=%5.3f Omega=%8.5f dist=%5.3f mineig=%7.3f log10|dmu|=%6.2f "
                          "conv?%s" % \
                          (p, phisp[p,:].sum(), omega, np.linalg.norm(dxsp[p,:]), \
                           min(eigsp), math.log10(max(1.e-20, np.linalg.norm(dmu))), isconv))
                elif verbose > 1:
                    print("  -> phisp =", phisp[p,:], \
                          "phiT=%5.3f Omega=%8.5f dist=%5.3f mineig=%5.3f log10|dmu|=%6.2f %s" % \
                          (phisp[p,:].sum(), omega, np.linalg.norm(dxsp[p,:]), \
                           min(eigsp), math.log10(max(1.e-20, np.linalg.norm(dmu))), isconv))
                    if not isconv:
                        print("    NOT CONVERGED:")
                        print("    log(phi) =", np.log(phisp[p,:]))
                if verbose > 2:
                    muexsp = mu - np.log(phisp[p,:])
                    print("  -> muex =", muexsp)
                elif verbose > 3:
                    print("   m(eig) =", eigvsp[:,np.argmin(eigsp)], \
                          "mineig=%8.3f align=%6.3f" % (min(eigsp), align))
        if not isconv: # Stationary point invalid
            raise SolverException
    dist = np.zeros((phi.shape[0], phi.shape[0]))
    for p in range(phi.shape[0]):
        for q in range(phi.shape[0]):
            dist[p,q] = np.linalg.norm(xsp[p,:] - phi[q,:] / phi[q,:].sum())
    if verbose:
        with np.printoptions(formatter={'float': '{: 6.3f}'.format}, linewidth=240):
            if verbose > 1:
                print("xsp:, rank =", np.linalg.matrix_rank(xsp, tol=1.e-3))
                print(xsp)
            print("sp-distances:, rank =", np.linalg.matrix_rank(dist, tol=1.e-3))
            print(dist)
    sdist = symm_dist_matrix(xsp)
    success = (all(dist[p,p] <= dist[p,:].min() and dist[p,p] <= maxdist_success \
                   for p in range(dist.shape[0])) and \
               (np.linalg.matrix_rank(sdist, tol=1.e-6) == xsp.shape[0] or sdist.shape == (1,1)))
    if verbose:
        print("LANDSCAPE TARGETS:: Success?", success)
    return {'phisp' : phisp,
            'dxsp' : dxsp,
            'omegasp' : omegasp,
            'dmusp' : dmusp,
            'mineigsp' : mineigsp,
            'mineigvsp' : mineigvsp,
            'success' : success}
