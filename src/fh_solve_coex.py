import math, argparse, gzip, pickle, random
import numpy as np
import networkx as nx
from contextlib import ExitStack

from epsilon_matrix import gen_symm_gaussian_noise
from fh import define_free_energy_fh
from fh_program import ProgramException
from fh_test import test_landscape_dilute, test_landscape_targets, SolverException
from fh_coexistence import guess_mu_coex, solve_coexistence, solve_coexistence_iter, \
    solve_coexistence_deps_iter, test_negative, find_parent_phase, solve_mole_fractions, \
    find_parent_phase_random_mole_fractions, solve_mole_fractions_systematically_remove_phases

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to pre-calculated results .p.gz file")
    parser.add_argument('--deps', action='store_true', help="perform epsilon optimization [False]")
    parser.add_argument('--negative-design', action='store_true', \
                        help="search for off-target phases [False]")
    parser.add_argument('--index', type=int, default=None, \
                        help="perform computations for specified target index only [None]")
    parser.add_argument('--examine-parent-phase', action='store_true', \
                        help="examine constraints on the parent phase [False]")
    parser.add_argument('--verbose', action='store_true', help="turn on verbose output [False]")
    parser.add_argument('--scale-epsilon', type=float, default=1., \
                        help="scale epsilon matrix by scalar [1.]")
    parser.add_argument('--noise', type=float, default=0., \
                        help="add Gaussian noise with specified standard deviation to epsilon [0.]")
    parser.add_argument('--output', type=str, default=None, \
                        help="output .p.gz path to save results [None]")
    clargs = parser.parse_args()

    with ExitStack() as stack:
        if clargs.output is not None:
            foutput = stack.enter_context(gzip.open(clargs.output, 'wb'))

        with gzip.open(clargs.path, 'rb') as f:
            while True:
                try:
                    results = pickle.load(f)
                except EOFError:
                    break
                if clargs.index is not None and results['target_index'] != clargs.index:
                    continue

                print("target index =", results['target_index'])

                N = results['targets'].shape[1]
                x = np.zeros(results['targets'].shape)
                for alpha in range(results['targets'].shape[0]):
                    x[alpha,:] = results['targets'][alpha,:] / results['targets'][alpha,:].sum()
                print("target compositions:")
                with np.printoptions(formatter={'float': '{: 5.3f}'.format}, linewidth=240):
                    print(x)
                phi_target = results['phiTa'] * x

                L = results['L'] * np.ones(N)
                mu_coex_guess = results['mu'].vector()
                eps_r = clargs.scale_epsilon * results['eps'].low_rank_approx(results['r'])[2]

                if clargs.noise > 0:
                    eps_r += gen_symm_gaussian_noise(results['eps'], clargs.noise)

                try:
                    phi0 = results['test_results_dilute']['phisp']
                    phi_est = results['test_results_targets']['phisp']
                    if not clargs.deps:
                        mu_coex, omega_coex = solve_coexistence_iter(L, eps_r, mu_coex_guess, phi0, \
                                                                     phi_est, verbose=clargs.verbose)
                    else:
                        mu_coex, omega_coex = solve_coexistence(L, eps_r, mu_coex_guess, phi0, \
                                                                phi_est, verbose=clargs.verbose)
                        eps_r, mu_coex, omega_coex = \
                            solve_coexistence_deps_iter(L, eps_r, mu_coex, phi0, phi_est, \
                                                        verbose=clargs.verbose)
                except (KeyError, SolverException, np.linalg.LinAlgError) as e:
                    print(e)
                    print("Skipping...\n")
                    continue

                F = define_free_energy_fh(L, eps_r)
                randoffset = 0.
                test_results_dilute = test_landscape_dilute(F, mu_coex, phi0, randoffset=randoffset, \
                                                            verbose=clargs.verbose)
                test_results_targets = test_landscape_targets(F, mu_coex, phi_est, \
                                                              randoffset=randoffset, \
                                                              verbose=clargs.verbose)
                with np.printoptions(formatter={'float': '{: 5.3f}'.format}, linewidth=240):
                    print("mu_coex =", mu_coex)
                print("<phi0_coex> =", np.mean(test_results_dilute['phisp']))
                omega_std = np.std([test_results_dilute['omegasp']] + \
                                   [test_results_targets['omegasp'][alpha] \
                                    for alpha in range(len(results['targets']))])
                print("COEXISTENCE TEST:", omega_std < 1.e-12, "; std(omega_sp) =", omega_std)
                print("LANDSCAPE TESTS:", test_results_dilute['success'], \
                      test_results_targets['success'])
                with np.printoptions(formatter={'float': '{: 5.3f}'.format}, linewidth=240):
                    for alpha in range(len(results['targets'])):
                        x_alpha_sp = (test_results_targets['phisp'][alpha,:] \
                                   / test_results_targets['phisp'][alpha,:].sum())
                        print("  x_%03d =" % (alpha + 1), x_alpha_sp, \
                              "phiT=%5.3f" % test_results_targets['phisp'][alpha,:].sum(), \
                              "dist=%5.3f" % np.linalg.norm(x_alpha_sp - x[alpha,:]), \
                              "Omega=%10.7f" % test_results_targets['omegasp'][alpha])

                # Does a parent phase exist? Find the one that minimizes the inf-norm of the
                #   mole fractions:
                print("PARENT PHASE TESTS (Deterministic):")
                phi_parent, mole_fractions = \
                    find_parent_phase(test_results_dilute['phisp'], test_results_targets['phisp'], \
                                      verbose=clargs.verbose)

                omega_parent = F[0](phi_parent) - np.dot(mu_coex, phi_parent)
                with np.printoptions(formatter={'float': '{: 5.2f}'.format}, linewidth=240):
                    print("  parent  =", phi_parent)
                    print("  molfrac =", mole_fractions)
                print("  omega_parent - omega_coex = %g" % (omega_parent - omega_coex))
                if clargs.examine_parent_phase:
                    solve_mole_fractions_systematically_remove_phases(test_results_dilute['phisp'], \
                                                                      test_results_targets['phisp'], \
                                                                      phi_parent, \
                                                                      verbose=clargs.verbose)
                    # Examine a few parent phases found by randomly generating mole fractions:
                    for trial in range(4):
                        print("PARENT PHASE TESTS (Random trial %d):" % trial)
                        phi_parent, mole_fractions = \
                            find_parent_phase_random_mole_fractions(test_results_dilute['phisp'], \
                                                                    test_results_targets['phisp'], \
                                                                    verbose=clargs.verbose)
                        omega_parent = F[0](phi_parent) - np.dot(mu_coex, phi_parent)
                        print("omega_parent - omega_coex = %g" % (omega_parent - omega_coex))
                        solve_mole_fractions_systematically_remove_phases(test_results_dilute['phisp'],\
                                                                        test_results_targets['phisp'], \
                                                                        phi_parent, \
                                                                        verbose=clargs.verbose)

                if clargs.negative_design:
                    negdesign_results = test_negative(L, eps_r, mu_coex, phi0, phi_est, nsamples=1000, \
                                                      verbose=clargs.verbose)
                    if negdesign_results['success']:
                        print("NEGATIVE DESIGN: Pass")
                    else:
                        if negdesign_results['nmorestable'] > 0:
                            print("NEGATIVE DESIGN: Fail (More stable: n = %d)" % \
                                  negdesign_results['nmorestable'])
                        else:
                            print("NEGATIVE DESIGN: Fail (Equally stable: n = %d)" % \
                                  negdesign_results['nequallystable'])
                        with np.printoptions(formatter={'float': '{: 6.3f}'.format}, linewidth=240):
                            for b in negdesign_results['basins']:
                                print(" ", b[0], b[1])
                else:
                    negdesign_results = None

                print()

                if clargs.output is not None:
                    results.update({'eps' : eps_r, \
                                    'mu_coex' : mu_coex, \
                                    'omega_coex' : omega_coex, \
                                    'test_results_dilute' : test_results_dilute, \
                                    'test_results_targets' : test_results_targets, \
                                    'omega_std' : omega_std, \
                                    'negdesign_results' : negdesign_results})
                    pickle.dump(results, foutput)
                    foutput.flush()
