import math, argparse, random, time, gzip, pickle
import numpy as np
import cvxpy as cp
import scipy.optimize

from fh import define_free_energy_fh, estimate_dilute_phi
from fh_test import test_landscape_dilute, test_landscape_targets
from epsilon_matrix import EpsilonMatrix, MuVector

from cfh import design_inequalities_fast, design_inequalities_muvar_fast

class ProgramException(Exception):
    pass

# Routines to estimate phi^(alpha):

def estimate_phiTa(x, L, phi0):
    # Estimate the total volume fraction, phi_T, for each condensed phase by assuming that the pressure
    #   at coexistence is exactly zero (and thus the dilute phase has total volume fraction ~= zero).
    phiTa = np.zeros(x.shape[0])
    def gen_diff(alpha):
        A = sum(x[alpha,i] * math.log(x[alpha,i] / phi0[i]) / L[i] if x[alpha,i] > 0. else 0. \
                for i in range(x.shape[1]))
        def diff(phiTa):
            if phiTa >= 1. or phiTa <= 0.:
                return 1.e9
            return -math.log(1. - phiTa) - np.dot(x[alpha,:], (1. - 1./L) * phiTa) \
                + 0.5 * phiTa * (-np.dot(x[alpha,:], math.log(phiTa) / L) - A + math.log(1. - phiTa))
        return diff
    for alpha in range(x.shape[0]):
        diff = gen_diff(alpha)
        Lavg = sum(x[alpha,i] / L[i] for i in range(x.shape[1]))
        phi_crit = 1. / (Lavg**(-0.5) + 1.)
        phiTa[alpha] = scipy.optimize.root_scalar(diff, bracket=[phi_crit, 1.-1.e-6], \
                                                  method='brentq').root
    return phiTa    

def estimate_phiTa_mu(x, L, mu):
    # Estimate the total volume fraction, phi_T, for each condensed phase by assuming that the pressure
    #   at coexistence is exactly zero (and thus the dilute phase has total volume fraction ~= zero).
    phiTa = np.zeros(x.shape[0])
    def gen_diff(alpha):
        Lavg = sum(x[alpha,i] / L[i] for i in range(x.shape[1]))
        muavg = sum(x[alpha,i] * mu[i] for i in range(x.shape[1]))
        A = (1. - Lavg - muavg \
             + sum(x[alpha,i] * math.log(x[alpha,i]) / L[i] if x[alpha,i] > 0. else 0. \
                   for i in range(x.shape[1])))
        def diff(phiTa):
            if phiTa >= 1. or phiTa <= 0.:
                return 1.e9
            return 2. * math.log(1. - phiTa) / phiTa + Lavg * math.log(phiTa) - math.log(1. - phiTa) + A
        return diff
    for alpha in range(x.shape[0]):
        diff = gen_diff(alpha)
        Lavg = sum(x[alpha,i] / L[i] for i in range(x.shape[1]))
        phi_crit = 1. / (Lavg**(-0.5) + 1.)
        phiTa[alpha] = scipy.optimize.root_scalar(diff, bracket=[phi_crit, 1.-1.e-6], \
                                                  method='brentq').root
    return phiTa

def estimate_phi_targets_composition(x, L, phi0, zeta):
    # Estimate the volume fractions in the condensed phases by computing phi_T^(alpha) for each phase
    #   using the approximation given above, assuming that the dilute phase has volume fractions phi_0,
    #   and the maximum excluded-component volume fractions as specified by zeta.
    phiTa = estimate_phiTa(x, L, phi0)
    phi = np.zeros(x.shape)
    for alpha in range(phi.shape[0]):
        card_alpha = np.count_nonzero(x[alpha,:])
        if card_alpha < x.shape[1]:
            phi_excl = zeta / (card_alpha * (x.shape[1] - card_alpha))
        else:
            phi_excl = 0.
        phi[alpha,:] = (x[alpha,:] + \
                        phi_excl * np.array([1. if x[alpha,k] == 0 else 0. for k in range(x.shape[1])]))
        phi[alpha,:] *= phiTa[alpha] / phi[alpha,:].sum()
    return phi

def estimate_phi_targets(phi, L, zeta, normalized=True):
    # Estimate the volume fractions in the condensed phases assuming that the maximum
    #   excluded-component volume fractions as specified by zeta.
    phiT = phi.sum(axis=1)
    phi_out = np.zeros(phi.shape)
    for alpha in range(phi.shape[0]):
        card_alpha = np.count_nonzero(phi[alpha,:])
        if card_alpha < phi.shape[1]:
            phi_excl_alpha = phiT[alpha] * zeta / (card_alpha * (phi.shape[1] - card_alpha))
        else:
            phi_excl_alpha = 0.
        phi_out[alpha,:] = (phi[alpha,:] + \
                            phi_excl_alpha * np.array([1. if phi[alpha,k] == 0 else 0. \
                                                       for k in range(phi.shape[1])]))
        if normalized:
            phi_out[alpha,:] *= phiT[alpha] / phi_out[alpha,:].sum()
    return phi_out

def calc_epsilon_target_means(phi, L):
    phiT = phi.sum(axis=1)
    cT = phiT - sum(phi[:,i] / L[i] for i in range(phi.shape[1]))
    eps_mean = (2. / phiT**2) * (np.log(1. - phiT) + cT)
    return eps_mean

# Definitions of FH design constraints:

def design_inequalities(x, L, phi0, zeta, eps, scale=True):
    if not scale:
        Xeq, yeq, Xin, yin = design_inequalities_fast(x, L, phi0, zeta, eps.index)
    else:
        phiTa = estimate_phiTa(x, L, phi0)
        x_scaled = np.multiply(phiTa, x.T).T
        Xeq, yeq, Xin, yin = design_inequalities_fast(x_scaled, L, phi0, zeta, eps.index)
    if yin.shape[0] > 0:
        return (Xeq @ eps.vec - yeq), (Xin @ eps.vec - yin)
    else: # Trivially satisfy the inequality constraints if there are none:
        return (Xeq @ eps.vec - yeq), -1.

def cone_definitions(x, L, phi0, zeta, eps, scaleL=True):
    phi = estimate_phi_targets_composition(x, L, phi0, zeta)
    phiT0 = phi0.sum()
    phiTa = estimate_phiTa(x, L, phi0)
    if scaleL:
        LL = np.outer(np.sqrt(L), np.sqrt(L))
        dilute_cone = (np.diag(1. / phi0) + LL / (1. - phiT0) + cp.multiply(LL, eps.cp_matrix()))
    else:
        dilute_cone = (np.diag(1. / (L * phi0)) + 1. / (1. - phiT0) + eps.cp_matrix())
    cones = [dilute_cone]
    for alpha in range(x.shape[0]):
        if scaleL:
            cone_alpha = (np.diag(1. / phi[alpha,:]) + LL / (1. - phiTa[alpha]) \
                          + cp.multiply(LL, eps.cp_matrix()))
        else:
            cone_alpha = np.diag(1. / (L * phi[alpha,:])) + 1. / (1. - phiTa[alpha]) + eps.cp_matrix()
        cones.append(cone_alpha)
    return cones

def on_diagonal_eps_constraint(x, L, phi0, eps):
    phiTa = estimate_phiTa(x, L, phi0)
    phi = x.copy()
    for a in range(x.shape[0]):
        phi[a,:] *= phiTa[a]
    eps_means = calc_epsilon_target_means(phi, L)
    eps_diag = eps.cp_target_diag()
    return [(cp.min(eps_diag[a]) >= eps_means[a]) for a in range(phi.shape[0])]

def design_inequalities_muvar(phi, L, zeta, eps, mu):
    Xeq, Yeq, yeq, Xin, Yin, yin = design_inequalities_muvar_fast(phi, L, zeta, eps.index)
    if yin.shape[0] > 0:
        return (Xeq @ eps.vec + Yeq @ mu.vec + yeq), (-(Xin @ eps.vec + Yin @ mu.vec + yin))
    else: # Trivially satisfy the inequality constraints if there are none:
        return (Xeq @ eps.vec + Yeq @ mu.vec + yeq), -1.

def design_inequalities_muvar_P(phi, L, zeta, eps, mu):
    # Separate the equal pressure conditions (last n rows in Xeq) from the equal chem. pot. conditions:
    Xeq, Yeq, yeq, Xin, Yin, yin = design_inequalities_muvar_fast(phi, L, zeta, eps.index)
    n = phi.shape[0]
    Neq = Xeq.shape[0]
    if yin.shape[0] > 0:
        return ((Xeq[:Neq-n,:] @ eps.vec + Yeq[:Neq-n] @ mu.vec + yeq[:Neq-n]), \
                (Xeq[Neq-n:,:] @ eps.vec + Yeq[Neq-n:] @ mu.vec + yeq[Neq-n:]), \
                (-(Xin @ eps.vec + Yin @ mu.vec + yin)))
    else: # Trivially satisfy the inequality constraints if there are none:
        return ((Xeq[:Neq-n,:] @ eps.vec + Yeq[:Neq-n] @ mu.vec + yeq[:Neq-n]), \
                (Xeq[Neq-n:,:] @ eps.vec + Yeq[Neq-n:] @ mu.vec + yeq[Neq-n:]), \
                -1.)

def cone_definitions_muvar(phi, L, zeta, eps, scaleL=True):
    phi_actual = estimate_phi_targets(phi, L, zeta, normalized=False)
    phiTa = phi.sum(axis=1)
    LL = np.outer(np.sqrt(L), np.sqrt(L))
    cones = []
    for alpha in range(phi.shape[0]):
        if scaleL:
            cone_alpha = (np.diag(1. / phi_actual[alpha]) + LL / (1. - phiTa[alpha]) \
                          + cp.multiply(LL, eps.cp_matrix()))
        else:
            cone_alpha = (np.diag(1. / (L * phi_actual[alpha])) + 1. / (1. - phiTa[alpha]) \
                          + eps.cp_matrix())
        cones.append(cone_alpha)
    return cones

def on_diagonal_eps_constraint_muvar(phi, L, eps):
    eps_means = calc_epsilon_target_means(phi, L)
    eps_diag = eps.cp_target_diag()
    return [(cp.min(eps_diag[a]) >= eps_means[a]) for a in range(phi.shape[0])]

### Base solvers:

def get_solver_options(solver_options):
    default_solver_options = {'solver':cp.SCS, 'eps':1.e-6, 'max_iters':1000000}
    default_solver_options.update(solver_options)
    return default_solver_options

base_solvers = {}

def solve_sdp_muvar_obj_ridge(phi, L, zeta, aux_constraints, nonpos=False, \
                              mineig_value=1., mu_norm_weight=1., mu_norm=2, ridge_norm='froD', \
                              ondiag_constraint=True, solver_options={}):
    # Solve the minimum-Euclidean-norm (inspired by negative design) SDP problem for epsilon and mu
    #   with inequality constraints on the excluded components as well as explicit constraints on the
    #   second-derivative matrices.  In this version, phi is specified exactly, mu is an independent
    #   variable, and phi0 is estimated.
    N = phi.shape[1]
    eps = EpsilonMatrix(phi, nonpos=nonpos)
    mu = MuVector(phi)
    designeq, designin = design_inequalities_muvar(phi, L, zeta, eps, mu)
    cones = cone_definitions_muvar(phi, L, zeta, eps)
    phiTa_mean = np.mean(np.sum(phi, axis=1))
    log_phiT0 = cp.log_sum_exp(cp.multiply(L, mu.vec) + L - np.ones(N)) # Ideal gas approx. for phiT0.
    phi_crit = 1. / (np.mean(1. / L)**(-0.5) + 1.) # Equimolar critical total volume fraction.
    mu_diff_matrix = 0.5 * (mu.cp_row_diff_matrix() + mu.cp_row_diff_matrix().T)
    domega_matrix = (eps.cp_matrix() - mu_diff_matrix / phiTa_mean) # Estimated d(Omega) for random phi.
    if ridge_norm == 'fro':
        objective = (cp.norm(domega_matrix, "fro") \
                     + mu_norm_weight * cp.norm(cp.multiply(L, mu.vec - cp.sum(mu.vec) / N), mu_norm))
    elif ridge_norm == 'froD':
        w = np.array([[math.sqrt(2.) if i == j else 1. for i in range(N)] for j in range(N)])
        objective = (cp.norm(cp.multiply(w, domega_matrix), "fro") \
                     + mu_norm_weight * cp.norm(cp.multiply(L, mu.vec - cp.sum(mu.vec) / N), mu_norm))
    elif ridge_norm == 'inf':
        objective = (cp.norm(cp.reshape(domega_matrix, (1, N*N)), "inf") \
                     + mu_norm_weight * cp.norm(cp.multiply(L, mu.vec - cp.sum(mu.vec) / N), mu_norm))
    elif ridge_norm == 'infD':
        w = np.array([[2. if i == j else 1. for i in range(N)] for j in range(N)])
        objective = (cp.norm(cp.reshape(cp.multiply(w, domega_matrix), (1, N*N)), "inf") \
                     + mu_norm_weight * cp.norm(cp.multiply(L, mu.vec - cp.sum(mu.vec) / N), mu_norm))
    else: raise Exception("unknown ridge norm")
    constraints = [(designeq == 0), (designin <= 0), (log_phiT0 <= math.log(0.9 * phi_crit))]
    constraints += [(c >> np.eye(N) * mineig_value) for c in cones]
    if ondiag_constraint: constraints += on_diagonal_eps_constraint_muvar(phi, L, eps)
    if aux_constraints is not None: constraints += aux_constraints(eps)
    solver_options = get_solver_options(solver_options)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(**solver_options)
    return {'eps' : eps, 'mu' : mu, 'problem' : problem, \
            'objective' : {'objective' : objective}, 'constraints' : constraints}
base_solvers['ridge'] = solve_sdp_muvar_obj_ridge

def solve_sdp_muvar_obj_similarity(phi, L, zeta, eps_ref, aux_constraints, nonpos=False, \
                                   mineig_value=1., similarity_norm="fro", \
                                   ridge_weight=0., mu_norm_weight=1., solver_options={}):
    # Solve the minimum-eigenvalue SDP problem with inequality constraints on the excluded components
    #   as well as explicit constraints on the second-derivative matrices.  Include an additional
    #   penalty on the maximum value of epsilon if above a certain threshold in the objective function.
    #   In this version, phi is specified exactly, mu is an independent variable, and phi0 is estimated.
    N = phi.shape[1]
    eps = EpsilonMatrix(phi, nonpos=nonpos)
    mu = MuVector(phi)
    designeq, designin = design_inequalities_muvar(phi, L, zeta, eps, mu)
    cones = cone_definitions_muvar(phi, L, zeta, eps)
    phiTa_mean = np.mean(np.sum(phi, axis=1))
    log_phiT0 = cp.log_sum_exp(cp.multiply(L, mu.vec) + L - np.ones(N)) # Ideal gas approx. for phiT0.
    phi_crit = 1. / (np.mean(1. / L)**(-0.5) + 1.) # Equimolar critical total volume fraction.
    mu_diff_matrix = 0.5 * (mu.cp_row_diff_matrix() + mu.cp_row_diff_matrix().T)
    domega_matrix = (eps.cp_matrix() - mu_diff_matrix / phiTa_mean) # Estimated d(Omega) for random phi.
    w = np.array([[math.sqrt(2.) if i == j else 1. for i in range(N)] for j in range(N)])
    ridge = cp.norm(cp.multiply(w, domega_matrix), "fro")
    if similarity_norm == 1:
        dissimilarity = cp.norm(cp.reshape(eps.cp_matrix() - eps_ref, (N*N)), 1)
    else:
        dissimilarity = cp.norm(eps.cp_matrix() - eps_ref, similarity_norm)
    objective = (dissimilarity \
                 + mu_norm_weight * cp.norm(cp.multiply(L, mu.vec - cp.sum(mu.vec) / N)) \
                 + ridge_weight * ridge)
    constraints = [(designeq == 0), (designin <= 0), (log_phiT0 <= math.log(0.9 * phi_crit))]
    constraints += [(c >> np.eye(N) * mineig_value) for c in cones]
    if aux_constraints is not None: constraints += aux_constraints(eps)
    solver_options = get_solver_options(solver_options)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(**solver_options)
    return {'eps' : eps, 'mu' : mu, 'problem' : problem, \
            'objective' : {'objective' : objective, 'dissimilarity' : dissimilarity, 'ridge' : ridge}, \
            'constraints' : constraints}
base_solvers['similarity'] = solve_sdp_muvar_obj_similarity

### Inherited solvers:

constraint_choices = {'' : None}
def gen_solver(base, constraint):
    def solver(*args, **kwargs):
        return base_solvers[base](*args, constraint_choices[constraint], **kwargs)
    return solver
solvers = {}
for bs in base_solvers:
    for c in constraint_choices:
        solvers[bs + c] = gen_solver(bs, c)
