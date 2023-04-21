import math
import numpy as np
import scipy.optimize

def define_free_energy_fh(L, eps):
    # Definition of Flory--Huggins (or regular solution if L==1) model.
    # We use an NxN matrix of interaction energies epsilon instead of the Flory chi parameter; in
    #   general, epsilon_ii != 0.  We assume that epsilon_is = epsilon_ss = 0, where s is the solvent.
    def F(phi, verbose=False):
        if phi.sum() >= 1. or np.any(phi <= 0.):
            return np.inf
        return (np.dot(phi / L, np.log(phi)) + (1. - phi.sum()) * math.log(1. - phi.sum()) \
                + 0.5 * np.dot(phi, np.dot(eps, phi)))
    def mu(phi):
        return np.log(phi) / L - math.log(1. - phi.sum()) - (1. - 1./L) + np.dot(eps, phi)
    def dmudphi(phi):
        return np.diag(1. / (L*phi)) + 1. / (1. - phi.sum()) + eps
    def P(phi):
        return (-math.log(1. - phi.sum()) - ((1. - 1./L) * phi).sum() \
                + 0.5 * np.dot(phi, np.dot(eps, phi)))
    return F, mu, dmudphi, P, {'L' : L, 'eps' : eps}

def estimate_dilute_phi(L, mu):
    # Estimate phi0 from mu assuming that phi0 ~= 0.
    return np.exp(L * mu + L - 1.)

def calculate_chi(eps):
    # Calculate the (N+1)x(N+1) Flory--Huggins chi matrix from the NxN epsilon matrix.
    # Column N+1 is the solvent.
    chi = np.zeros((eps.shape[0]+1, eps.shape[1]+1))
    for i in range(eps.shape[0]):
        for j in range(i + 1, eps.shape[1]):
            chi[i,j] = chi[j,i] = eps[i,j] - 0.5 * (eps[i,i] + eps[j,j])
        chi[i,-1] = chi[-1,i] = -0.5 * eps[i,i]
    return chi

def solve_dmu(F, mu, phiinit, scale=1., gtol=1.e-12):
    # Solve dF/dphi = mu by LM.  Note that F is a function of phi,
    #    but we are using ln(phi) as the independent variable.
    def diff(lnphi):
        phi = np.exp(lnphi)
        if phi.sum() >= 1.:
            return np.inf * np.ones(mu.shape)
        return (scale * F[1](phi) - scale * mu)
    def J(lnphi):
        phi = np.exp(lnphi)
        if phi.sum() >= 1.:
            return np.inf * np.ones(mu.shape)
        phiI = np.diag(phi)
        return np.dot(scale * F[2](phi), phiI)
    lnphiinit = np.log(phiinit)
    sln = scipy.optimize.root(diff, lnphiinit, method='lm', jac=J, options={'factor':0.1, 'gtol':gtol})
    return np.exp(sln.x), (sln.fun / scale), (sln.success, sln.message)

def minimize_P(F, mu, phiinit, scale=1., gtol=1.e-10, maxiter=1e2):
    # Minimize Omega = F - mu*phi by trust region minimization.
    # Note that F is a function of phi, but we are using ln(phi) as the independent variable.
    def Omega(lnphi):
        phi = np.exp(lnphi)
        if phi.sum() >= 1.:
            return np.inf
        return (scale * F[0](phi) - np.dot(phi, scale * mu))
    def J(lnphi):
        phi = np.exp(lnphi)
        if phi.sum() >= 1.:
            return np.inf * np.ones(mu.shape)
        phiI = np.diag(phi)
        dmu = (scale * F[1](phi) - scale * mu)
        return np.dot(phiI, dmu)
    def H(lnphi):
        phi = np.exp(lnphi)
        if phi.sum() >= 1.:
            return np.inf * np.ones((mu.shape[0], mu.shape[0]))
        phiI = np.diag(phi)
        return (np.dot(phiI, np.diag(scale * F[1](phi) - scale * mu)) \
                + np.dot(phiI, np.dot(scale * F[2](phi), phiI)))
    lnphiinit = np.log(phiinit)
    sln = scipy.optimize.minimize(Omega, lnphiinit, method='trust-ncg', jac=J, hess=H, \
                                  options={'initial_trust_radius':0.01, 'max_trust_radius':0.1, \
                                           'gtol':gtol, 'maxiter':maxiter})
    return np.exp(sln.x), (sln.fun / scale), (sln.success, sln.message)

def eig_symm_safe(A, maxval=1.e9):
    # Return eigenvalues and eigenvectors of a symmetric matrix.
    # Exclude rows/columns with large entries on the diagonal.
    exclude = [i for i in range(A.shape[0]) if A[i,i] > maxval]
    if len(exclude) == 0:
        return np.linalg.eigh(A)
    else:
        AA = np.zeros((A.shape[0] - len(exclude), A.shape[1] - len(exclude)))
        ii = 0
        for i in range(A.shape[0]):
            if i in exclude: continue
            jj = 0
            for j in range(A.shape[1]):
                if j in exclude: continue
                AA[ii,jj] = A[i,j]
                jj += 1
            ii += 1
        eigvals_AA, eigvecs_AA = np.linalg.eigh(AA)
        eigvals_A = np.array([eigvals_AA[i] for i in range(len(eigvals_AA))] + [np.inf] * len(exclude))
        eigvecs_A = np.zeros(A.shape)
        for j in range(eigvecs_AA.shape[1]):
            ii = 0
            for i in range(A.shape[0]):
                if i in exclude: continue
                eigvecs_A[i,j] = eigvecs_AA[ii,j]
                ii += 1
        for i in range(len(exclude)):
            eigvecs_A[exclude[i],-len(exclude)+i] = 1.
        return eigvals_A, eigvecs_A

def eigvals_symm_safe(A, maxval=1.e9):
    # Return eigenvalues of a symmetric matrix. Exclude rows/columns with large entries on the diagonal.
    exclude = [i for i in range(A.shape[0]) if A[i,i] > maxval]
    if len(exclude) == 0:
        return np.linalg.eigvalsh(A)
    else:
        AA = np.zeros((A.shape[0] - len(exclude), A.shape[1] - len(exclude)))
        ii = 0
        for i in range(A.shape[0]):
            if i in exclude: continue
            jj = 0
            for j in range(A.shape[1]):
                if j in exclude: continue
                AA[ii,jj] = A[i,j]
                jj += 1
            ii += 1
        eigvals_AA = np.linalg.eigvalsh(AA)
        return np.array([eigvals_AA[i] for i in range(len(eigvals_AA))] + [np.inf] * len(exclude))

def find_sp(F, mu, phiinit, minimize_P_params={}, solve_dmu_params={}, verbose=0):
    # Find a stationary point of F starting from phiinit.
    # This routine first attempts minimization and then uses root-finding for the derivative.
    def Omega(phi):
        return F[0](phi) - np.dot(phi, mu)
    if verbose:
        with np.printoptions(formatter={'float': '{: 8.6f}'.format}, linewidth=240):
            print("  FIND_SP:: INIT: phi =", phiinit)
    phi_min, Omega_min, status_min = minimize_P(F, mu, phiinit, **minimize_P_params)
    if verbose:
        print("  FIND_SP::  MIN: phiT=%5.3f, %5s: %s" % (phi_min.sum(), status_min[0], status_min[1]))
        with np.printoptions(formatter={'float': '{: 8.6f}'.format}, linewidth=240):
            print("                  phi =", phi_min)
    phi_dmu, dmu_dmu, status_dmu = solve_dmu(F, mu, phi_min, **solve_dmu_params)
    Omega_dmu = Omega(phi_dmu)
    d2sp = F[2](phi_dmu)
    eigsp = eigvals_symm_safe(d2sp)
    if verbose:
        print("  FIND_SP::  DMU: phiT=%5.3f, %5s: %s" % (phi_dmu.sum(), status_dmu[0], status_dmu[1]))
        print("  FIND_SP:: |phi_min - phi_init| = %g -> |phi_dmu - phi_min| = %g" % \
              (np.linalg.norm(phi_min - phiinit), np.linalg.norm(phi_dmu - phi_min)))
        print("  FIND_SP:: |dmu| = %g" % np.linalg.norm(dmu_dmu))
        print("  FIND_SP:: Omega_dmu = %g (dOmega = %g)" % (Omega_min, Omega_dmu))
        print("  FIND_SP:: cond(d2F) = %g" % np.linalg.cond(d2sp))
    success = ((np.linalg.norm(dmu_dmu) <= 1.e-9) and \
               eigsp.min() > 0. and \
               phi_dmu.min() > 0. and \
               phi_dmu.sum() < 1.)
    return phi_dmu, Omega_dmu, dmu_dmu, success
