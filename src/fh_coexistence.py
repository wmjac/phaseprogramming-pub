import math, argparse, random
import numpy as np
import scipy.optimize
import cvxpy as cp
from itertools import combinations

from fh import define_free_energy_fh, find_sp
from fh_test import SolverException

def guess_mu_coex(L, eps, phi0, phitargets):
    # Determine an initial estimate of mu by fitting a hyperplane to the minima of the free-energy
    # landscape.  NOTE: This could be improved by constructing the (regularized) hyperplane that
    # intersects F at phi0 and each of the phitargets.
    F = define_free_energy_fh(L, eps)
    F_0 = F[0](phi0)
    F_t = np.array([F[0](phitargets[p,:]) for p in range(phitargets.shape[0])])
    mu_guess = np.mean([(F_t[p] - F_0) / (phitargets[p,:].sum() - phi0.sum()) \
                        for p in range(phitargets.shape[0])]) * np.ones(phitargets.shape[1])
    return mu_guess

def solve_coexistence(L, eps, mu_guess, phi0, phitargets, maxdist_success=0.1, \
                      reg=1.e-1, mureg=False, verbose=False):
    # Attempt to solve for coexistence among a dilute phase (phi0) and n target phases (phitargets).
    # The solution is carried out in two steps.  First, we tune the average mu to minimize the
    # grand-potential differences.  Second, we tune the individual elements of mu to minimize the
    # grand-potential differences, subject to the regularization condition that the condensed-phase
    # compositions are as close to the target compositions as possible (regularization parameter: reg).
    # Alternatively, if mureg=True, we regularize by minimizing the standard deviation of the mu vector.
    # The coexistence solution must result in stationary points within the specified tolerance
    # (maxdist_success) of phi0 and phitargets.
    # NOTES: The parameter maxdist_success should match what is used in fh_test.py in order to
    #   guarantee that a solution determined from this routine will pass the landscape tests.  Also
    #   note that the regularization parameter, reg, is crucial for ensuring that the domegan routine
    #   finds a hyperplane touching stationary points near the input phitargets when the system
    #   is underdetermined; if it is too small, then this step may stray far from the target phisp.
    F = define_free_energy_fh(L, eps)
    def validate_mu(mu):
        phisp_0, omega_0, dmu_0, isconv = find_sp(F, mu, phi0)
        if not isconv or math.fabs(phisp_0.sum() - phi0.sum()) > maxdist_success:
            return False
        for a in range(phitargets.shape[0]):
            phisp_a, omega_a, dmu_a, isconv = find_sp(F, mu, phitargets[a,:])
            if not isconv or \
               np.linalg.norm(phisp_a / phisp_a.sum() - phitargets[a,:] / phitargets[a,:].sum()) \
               > maxdist_success: # * math.sqrt(phitargets.shape[0]):
                return False
        return True
    # Step 1: Adjust the mean of mu_guess to minimize the grand-potential differences.
    def domega1(dmu, verbose=False):
        mu = mu_guess + dmu * np.ones(mu_guess.shape)
        phisp_0, omega_0, dmu_0, isconv = find_sp(F, mu, phi0)
        if verbose: print('  d', phisp_0)
        if not isconv:
            return 1.e9 * np.ones(phitargets.shape[0])
        domega = np.zeros(phitargets.shape[0])
        for a in range(phitargets.shape[0]):
            phisp_a, omega_a, dmu_a, isconv = find_sp(F, mu, phitargets[a,:])
            if isconv:
                domega[a] = omega_a - omega_0
                if verbose: print(' ', a, phisp_a)
            else:
                domega[a] = 1.e9
        if verbose:
            print("  log10(|resid|):", np.log10(np.fabs(domega)))
        return domega
    if verbose:
        with np.printoptions(formatter={'float': '{: 7.4f}'.format}, linewidth=240):
            omega = find_sp(F, mu_guess, phi0)[1]
            print("coex --> initial guess:")
            print("  mu =", mu_guess)
            print("  omega =", omega)
            domega1(0, verbose=True)
    if verbose: print("coex --> mu 1:")
    with np.printoptions(formatter={'float': '{: 7.4f}'.format}, linewidth=240):
        dmu = scipy.optimize.leastsq(domega1, 0.)[0]
        mu_guess += dmu * np.ones(mu_guess.shape)
        if verbose:
            omega = find_sp(F, mu_guess, phi0)[1]
            print("  mu =", mu_guess)
            print("  omega =", omega)
            domega1(dmu, verbose=True)
    if not validate_mu(mu_guess):
        raise SolverException("first step of coexistence solver failed")
    # Step 2: Perform the full nonlinear optimization on mu, subject to regularization, starting
    #         from the updated mu_guess.  The regularization type is specified by mureg=True/False.
    def domegan(mu, verbose=False):
        phisp_0, omega_0, dmu, isconv = find_sp(F, mu, phi0)
        if verbose: print('  d', phisp_0)
        if not isconv:
            return 1.e9 * np.ones(phitargets.shape[0] * (1 + phitargets.shape[1]))
        resid = np.zeros(phitargets.shape[0] * (1 + phitargets.shape[1]))
        for a in range(phitargets.shape[0]):
            phisp_a, omega_a, dmu, isconv = find_sp(F, mu, phitargets[a,:])
            if isconv:
                resid[a] = omega_a - omega_0
                resid[phitargets.shape[0]+a*phitargets.shape[1]:\
                      phitargets.shape[0]+(a+1)*phitargets.shape[1]] \
                      = reg * (phisp_a / phisp_a.sum() - phitargets[a,:] / phitargets[a,:].sum())
                if verbose: print(' ', a, phisp_a)
            else:
                resid[a] = 1.e9
        if verbose:
            print("  log10(|resid[:ntarget]|):", np.log10(np.fabs(resid[:phitargets.shape[0]])))
            if reg > 0:
                print("  log10(|resid[ntarget:]|):", np.log10(np.fabs(resid[phitargets.shape[0]:])))
        if reg > 0:
            return resid
        else:
            return resid[:phitargets.shape[0]]
    def domegan_mureg(mu, verbose=False):
        phisp_0, omega_0, dmu, isconv = find_sp(F, mu, phi0)
        if verbose: print('d', phisp_0)
        if not isconv:
            return 1.e9 * np.ones(phitargets.shape[0] + 1)
        resid = np.zeros(phitargets.shape[0] + 1)
        for a in range(phitargets.shape[0]):
            phisp_a, omega_a, dmu, isconv = find_sp(F, mu, phitargets[a,:])
            if isconv:
                resid[a] = omega_a - omega_0
            else:
                resid[a] = 1.e9
        resid[-1] = np.std(mu)
        if verbose:
            print("log10(|resid[:ntarget]|):", np.log10(np.fabs(resid[:phitargets.shape[0]])))
            if reg > 0:
                print("log10(|resid[-1]|):", np.log10(np.fabs(resid[-1])))
        if reg > 0:
            return resid
        else:
            return resid[:phitargets.shape[0]]
    if verbose: print("coex --> mu n:")
    if mureg:
        domegan = domegan_mureg
    with np.printoptions(formatter={'float': '{: 7.4f}'.format}, linewidth=240):
        mu = scipy.optimize.leastsq(domegan, mu_guess)[0]
        omega = find_sp(F, mu, phi0)[1]
        if verbose:
            print("  mu =", mu)
            print("  omega =", omega)
            domegan(mu, verbose=True)
    if not validate_mu(mu):
        raise SolverException("second step of coexistence solver failed")
    return mu, omega

def solve_coexistence_iter(L, eps, mu_coex, phi0, phitargets, \
                           verbose=False, ntrials=10):
    # Perform iterative mu optimization to determine coexistence using solve_coexistence.
    # At each iteration, regularization is based on the previous solution for mu_coex and phisp_target.
    F = define_free_energy_fh(L, eps)
    phisp_old = phitargets.copy()
    for trial in range(ntrials):
        mu_coex, omega_coex = solve_coexistence(L, eps, mu_coex, phi0, phisp_old, \
                                                verbose=verbose)
        phisp_new = np.zeros(phisp_old.shape)
        for a in range(phitargets.shape[0]):
            phisp_a, omega_a, dmu, isconv = find_sp(F, mu_coex, phitargets[a,:])
            if not isconv:
                raise SolverException("iterative coexistence solver failed (find_sp not converged)")
            phisp_new[a,:] = phisp_a
        dphisp_norm = np.linalg.norm(phisp_new - phisp_old)
        if verbose:
            print("--> SOLVE DMU TRIAL %d:" % trial, dphisp_norm)
        phisp_old = phisp_new
        if dphisp_norm <= 1.e-15:
            break
    return mu_coex, omega_coex

def solve_coexistence_deps(L, eps0, mu0, phi0, phitargets, maxdist_success=0.1, verbose=False):
    # Attempt to solve for coexistence among a dilute phase (phi0) and n target phases (phitargets)
    # by tuning both mu and epsilon.  An initial guess of mu is required (and can be obtained from
    # solve_coexistence).  The solution is regularized by minimizing the norm of depsilon and dmu.
    Neps = eps0.shape[0] * (eps0.shape[0] + 1) // 2
    def validate_eps_mu(eps, mu):
        F = define_free_energy_fh(L, eps)
        phisp_0, omega_0, dmu_0, isconv = find_sp(F, mu, phi0)
        if not isconv or math.fabs(phisp_0.sum() - phi0.sum()) > maxdist_success:
            return False
        for a in range(phitargets.shape[0]):
            phisp_a, omega_a, dmu_a, isconv = find_sp(F, mu, phitargets[a,:])
            if not isconv or \
               np.linalg.norm(phisp_a / phisp_a.sum() - phitargets[a,:] / phitargets[a,:].sum()) \
               > maxdist_success: # * math.sqrt(phitargets.shape[0]):
                return False
        return True
    def domega_deps_dmu(y, reg=1.e-9, verbose=False):
        deps = np.zeros(eps0.shape)
        deps[np.triu_indices(deps.shape[0], k = 0)] = y[:Neps]
        deps += deps.T - np.diag(np.diag(deps))
        mu = mu0 + y[Neps:]
        F = define_free_energy_fh(L, eps0 + deps)
        phisp_0, omega_0, dmu, isconv = find_sp(F, mu, phi0)
        if verbose: print('d', phisp_0)
        if not isconv:
            return 1.e9 * np.ones(phitargets.shape[0] + y.shape[0])
        resid = np.zeros(phitargets.shape[0] + y.shape[0])
        for a in range(phitargets.shape[0]):
            phisp_a, omega_a, dmu, isconv = find_sp(F, mu, phitargets[a,:])
            if isconv:
                resid[a] = omega_a - omega_0
                if verbose: print(a, phisp_a)
            else:
                resid[a] = 1.e9
        resid[phitargets.shape[0]:] = reg * y
        if verbose:
            print("log10(|resid[:ntarget]|):", np.log10(np.fabs(resid[:phitargets.shape[0]])))
            if reg > 0:
                print("log10(|resid[ntarget:]|):", np.log10(np.fabs(resid[phitargets.shape[0]:])))
        if reg > 0:
            return resid
        else:
            return resid[:phitargets.shape[0]]
    if verbose: print("coex --> eps,mu:")
    y_guess = np.zeros(Neps + eps0.shape[0])
    with np.printoptions(formatter={'float': '{: 5.2f}'.format}, linewidth=240):
        y = scipy.optimize.leastsq(domega_deps_dmu, y_guess)[0]
        deps = np.zeros(eps0.shape)
        deps[np.triu_indices(deps.shape[0], k = 0)] = y[:Neps]
        deps += deps.T - np.diag(np.diag(deps))
        eps = eps0 + deps
        dmu = y[Neps:]
        mu = mu0 + dmu
        F = define_free_energy_fh(L, eps)
        omega = find_sp(F, mu, phi0)[1]
        if verbose:
            print("deps =")
            print(deps)
            print("mu =", mu)
            print("omega =", omega)
            domega_deps_dmu(y, verbose=True)
    if not validate_eps_mu(eps, mu):
        raise SolverException("eps,mu coexistence solver failed")
    return eps, mu, omega

def solve_coexistence_deps_iter(L, eps, mu_coex, phi0, phitargets, \
                                verbose=False, ntrials=10):
    # Perform iterative (eps,mu) optimization to determine coexistence using solve_coexistence_deps.
    # At each iteration, the regularization is based on the previous solution for epsilon.
    for trial in range(ntrials):
        eps_new, mu_coex, omega_coex = solve_coexistence_deps(L, eps, mu_coex, phi0, phitargets, \
                                                              verbose=verbose)
        deps_norm = np.linalg.norm(eps_new - eps)
        if verbose:
            print("--> SOLVE DEPS TRIAL %d:" % trial, deps_norm)
        eps = eps_new
        if deps_norm <= 1.e-15:
            break
    return eps, mu_coex, omega_coex

def sample_regular_simplex(N):
    # Sample an N-dimensional vector from within an N-dimensional regular simplex (N+1 vertices).
    r = [0.] + sorted(random.random() for i in range(N))
    return np.array([r[i+1] - r[i] for i in range(N)])

def test_negative(L, eps, mu, phi0, phitargets, nsamples=10000, phisp_tol=1.e-6, domega_tol=1.e-3, \
                  verbose=0):
    # Given a chemical potential mu for which the dilute (phi0) and target (phitargets) phases are in
    # coexistence, search for novel phases that are either more stable (omega < omega_coex) or equally
    # stable (omega ~= omega_coex) by minimizing the grand potential starting from random initial phi
    # points that are sampled uniformly from the concentration-space unit simplex.
    F = define_free_energy_fh(L, eps)
    phicoex = np.zeros((phitargets.shape[0] + 1, phitargets.shape[1]))
    phicoex0, omega_coex = find_sp(F, mu, phi0)[:2]
    phicoex[0,:] = phicoex0
    for a in range(phitargets.shape[0]):
        phicoex[a+1,:] = find_sp(F, mu, phitargets[a,:])[0]
    basins, basins_domega = [], []
    for sample in range(nsamples):
        phiinit = sample_regular_simplex(mu.shape[0])
        phisp, omega, dmu, isconv = find_sp(F, mu, phiinit)
        if isconv and omega <= omega_coex + domega_tol and \
           not any(np.linalg.norm(phisp - phicoex[a,:]) <= phisp_tol * math.sqrt(phicoex.shape[0]) \
                   for a in range(phicoex.shape[0])):
            if not any(np.linalg.norm(phisp - np.mean(np.array(b), axis=0)) \
                       <= phisp_tol * math.sqrt(phicoex.shape[0]) for b in basins):
                basins.append([phisp])
                basins_domega.append(omega - omega_coex)
            else:
                i = min(range(len(basins)), \
                        key=lambda i: np.linalg.norm(phisp - np.mean(np.array(basins[i]), axis=0)))
                basins[i].append(phisp)
                if omega - omega_coex < basins_domega[i]:
                    basins_domega[i] = omega - omega_coex
    basins = [(np.mean(np.array(basins[i]), axis=0), basins_domega[i]) for i in range(len(basins))]
    if verbose:
        for b in basins:
            print("  neg design: found basin:", b[0], b[1])
    nmorestable = sum(1 for i in range(len(basins)) if basins[i][1] < -domega_tol)
    nequallystable = len(basins) - nmorestable
    return {'success' : (len(basins) == 0), \
            'nmorestable' : nmorestable, \
            'nequallystable' : nequallystable, \
            'basins' : basins}

def sample_domega_distributions(L, eps, mu, phi0, nsamples=10000, bins=20):
    # Given epsilon and mu, calculate distributions for dOmega(phi) and the local minima of dOmega
    # using initial phi points that are sampled uniformly from the concentration-space unit simplex.
    F = define_free_energy_fh(L, eps)
    omega_coex = find_sp(F, mu, phi0)[1]
    omega_samples = []
    omega_minima_samples = []
    for sample in range(nsamples):
        phiinit = sample_regular_simplex(mu.shape[0])
        omega_samples.append(F[0](phiinit) - np.dot(mu, phiinit) - omega_coex)
        phisp, omega, dmu, isconv = find_sp(F, mu, phiinit)
        if isconv:
            omega_minima_samples.append(omega - omega_coex)
    return np.histogram(omega_samples, bins=bins), np.histogram(omega_minima_samples, bins=bins)

def find_parent_phase(phi0, phitargets, tol=0.001, norm='inf', verbose=False):
    # Find a suitable parent phase for a prescribed set of phases (if it exists) and calculate the
    # corresponding mole fractions.  Regularize by choosing the solution for which the target-phase
    # mole fractions are most similar to one another (according to the specified norm).
    n, N = phitargets.shape
    vec = cp.Variable(N + n)
    objective = cp.norm(vec[N:] - 1. / (n + 1), norm)
    constraints = [(cp.sum(vec[N:]) <= 1. - tol), \
                   (vec[N:] >= tol), \
                   (vec[N:] @ phitargets + (1. - cp.sum(vec[N:])) * phi0 == vec[:N])]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    phiparent = vec.value[:N]
    y = np.zeros(n + 1)
    y[:-1] = vec.value[N:]
    y[-1] = 1. - y.sum()
    if verbose:
        print("--> parent phase problem:", problem.status)
        with np.printoptions(formatter={'float': '{: 8.6f}'.format}, linewidth=240):
            print("    phi_parent =", phiparent)
            print("    fractions  =", y, y.sum())
            phiphases = np.zeros((n + 1, N))
            phiphases[:n,:] = phitargets
            phiphases[-1,:] = phi0
            print("    y * phi = phi_parent?", np.allclose(np.dot(y, phiphases), phiparent))
    if problem.status == cp.OPTIMAL:
        return phiparent, y # Returns phi_parent, mole fractions.

def find_parent_phase_random_mole_fractions(phi0, phitargets, verbose=False):
    # Find a suitable parent phase for a prescribed set of phases by randomly choosing the mole
    # fractions from the unit simplex and then solving for the parent-phase composition.
    n, N = phitargets.shape
    y = np.zeros(n + 1)
    y[:-1] = sample_regular_simplex(n)
    y[-1] = 1. - y[:-1].sum()
    phiphases = np.zeros((n + 1, N))
    phiphases[:n,:] = phitargets
    phiphases[-1,:] = phi0
    phiparent = np.dot(y, phiphases)
    if verbose:
        print("--> parent phase with random mole fractions:")
        with np.printoptions(formatter={'float': '{: 8.6f}'.format}, linewidth=240):
            print("    phi_parent =", phiparent)
            print("    fractions  =", y, y.sum())
    return phiparent, y # Returns phi_parent, mole fractions.

def solve_mole_fractions(phi0, phitargets, phiparent, excluded_phases=[], norm='inf', verbose=False):
    # Solve for the mole fractions that are consistent with the specified parent-phase composition
    # and that minimize the specified norm.  Mole fractions are set to zero for each of the specified
    # "excluded" phases.
    n, N = phitargets.shape
    vec = cp.Variable(n)
    objective = cp.norm(vec[N:] - 1. / (n + 1), norm)
    constraints = [(cp.sum(vec) <= 1.), \
                   (vec >= 0.), \
                   (vec @ phitargets + (1. - cp.sum(vec)) * phi0 == phiparent)]
    for i in excluded_phases:
        constraints.append((vec[i] == 0.))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    y = np.zeros(n + 1)
    y[:-1] = vec.value
    y[-1] = 1. - y.sum()
    if verbose:
        print("--> mole fraction problem:", problem.status)
        with np.printoptions(formatter={'float': '{: 8.6f}'.format}, linewidth=240):
            print("    phi_parent =", phiparent)
            print("    fractions  =", y, y.sum())
            phiphases = np.zeros((n + 1, N))
            phiphases[:n,:] = phitargets
            phiphases[-1,:] = phi0
            print("    y * phi = phi_parent?", np.allclose(np.dot(y, phiphases), phiparent))
    if problem.status == cp.OPTIMAL:
        return y # Returns mole fractions.

def solve_mole_fractions_systematically_remove_phases(phi0, phitargets, phiparent, verbose=False):
    # Attempt to solve for the mole fractions corresponding to the specified parent-phase composition
    # while systematically setting the mole fractions of select target phases to zero.
    n, N = phitargets.shape
    phiphases = np.zeros((n + 1, N))
    phiphases[:n,:] = phitargets
    phiphases[-1,:] = phi0
    if verbose:
        print("--> mole fraction systematic phase elimination:")
    successfully_eliminated = []
    for nremove in range(n + 1):
        success = False
        for z in combinations(range(n), r=nremove):
            y = solve_mole_fractions(phi0, phitargets, phiparent, excluded_phases=list(z))
            if y is not None and np.all(np.isfinite(y)):
                successfully_eliminated.append(z)
                success = True
                if not np.allclose(np.dot(y, phiphases), phiparent):
                    raise Exception
                if verbose:
                    with np.printoptions(formatter={'float': '{: 4.2f}'.format}, linewidth=240):
                        print("  Missing", z, \
                              "n_zero =", np.sum(1. for yy in y if math.fabs(yy) < 1.e-6), y)
        if not success:
            if verbose:
                print("  Cannot remove %d phases and still find a solution." % nremove)
            break
    return successfully_eliminated
