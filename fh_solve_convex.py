import math, argparse, gzip, pickle, random
import numpy as np
import networkx as nx
import cvxpy as cp
from contextlib import ExitStack

from fh import define_free_energy_fh, estimate_dilute_phi
from fh_program import estimate_phi_targets, ProgramException, solvers as base_solvers
from fh_test import test_landscape_dilute, test_landscape_targets, SolverException
from epsilon_matrix import count_close_elements

def load_targets(path):
    # Routine for loading targets from a text file:
    all_targets = []
    with open(path, 'r') as f:
        for line in f:
            if len(line) > 0 and line[0] != '#':
                if '[[' in line:
                    current = [line.strip().replace(' ',',')]
                else:
                    current.append(line.strip().replace(' ',','))
                if ']]' in line:
                    all_targets.append(eval('np.array(' + ','.join(current) + ')'))
    return all_targets

def cast_program_option_value(y):
    if y == 'inf':
        return y
    try:
        return float(y)
    except ValueError:
        return y

solvers = {}
for base_solver in base_solvers:
    solvers[base_solver] = (base_solvers[base_solver], {'nonpos' : False})
    solvers[base_solver + '-nonpos'] = (base_solvers[base_solver], {'nonpos' : True})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to targets file")
    parser.add_argument('solver', choices=list(solvers), help="solver name")
    parser.add_argument('--index', type=int, default=None, \
                        help="perform computations for specified target index only [None]")
    parser.add_argument('--skip-equiv-components', action='store_true', \
                        help="skip any target sets that contain equivalent components [False]")
    parser.add_argument('--phiTa', type=float, default=0.95, \
                        help="volume fraction of target phases [0.95]")
    parser.add_argument('--L', type=float, default=100., help="L [100.]")
    parser.add_argument('--zeta', type=float, default=1.e-2, help="excluded component zeta [1.e-2]")
    parser.add_argument('--input-path', type=str, default=None, \
                        help="path to input results .p.gz file for 'similarity' solver [None]")
    parser.add_argument('--input-index', type=int, default=None, \
                        help="index for input results .p.gz file for 'similarity' solver [None]")
    parser.add_argument('--solver-tol', type=float, default=1.e-4, help="solver tolerance [1.e-4]")
    parser.add_argument('--solver-maxiter', type=int, default=1000000, \
                        help="solver max iterations [1000000]")
    parser.add_argument('--options', type=str, default=None, help="convex program options [None]")
    parser.add_argument('--randomize-compositions', type=float, default=None, \
                        help="exponential scale factor for perturbing the compositions of the targets "
                             "away from equimolar [None]")
    parser.add_argument('--scale-component-compositions', type=str, default=None, \
                        help="scale the composition of each component by a specified amount in all "
                             "phases (format i:scalefactor_i,j:scalefactor_j,...) [None]")
    parser.add_argument('--verbose', action='store_true', help="turn on verbose output [False]")
    parser.add_argument('--output', type=str, default=None, \
                        help="output .p.gz path to save results [None]")
    parser.add_argument('--random-seed', type=int, default=None, help="set the PRNG seed [None]")
    clargs = parser.parse_args()

    if clargs.random_seed is not None:
        random.seed(clargs.random_seed)

    print("# solver:", clargs.solver)
    print("# phiTa:", clargs.phiTa)
    print("# L:", clargs.L)
    print("# zeta:", clargs.zeta)

    if clargs.options is not None:
        program_options = {kv.split(':')[0] : cast_program_option_value(kv.split(':')[1]) \
                           for kv in clargs.options.split(',')}
        for k,v in program_options.items():
            print("# convex program option '%s':" % k, v)
    else:
        program_options = {}

    print()

    all_targets = load_targets(clargs.path)
    print("Loaded %d targets\n" % len(all_targets))

    if clargs.index is not None and (clargs.index < 0 or clargs.index >= len(all_targets)):
        raise Exception("target index is out of bounds!")

    if 'similarity' in clargs.solver:
        if clargs.input_path is None or clargs.input_index is None:
            raise Exception("similarity solvers require both --input-path and --input-index options")
        eps_ref = None
        with gzip.open(clargs.input_path, 'rb') as f:
            while True:
                try:
                    input_results = pickle.load(f)
                except EOFError:
                    break
                if input_results['target_index'] == clargs.input_index:
                    eps_ref = input_results['eps'].matrix()
                    print("Loaded reference epsilon matrix =")
                    with np.printoptions(formatter={'float': '{: 5.3f}'.format}, linewidth=240):
                        print(eps_ref)
                    break
        if eps_ref is None:
            raise Exception("could not find target index in input results file")

    with ExitStack() as stack:
        if clargs.output is not None:
            foutput = stack.enter_context(gzip.open(clargs.output, 'wb'))

        for target_index in range(len(all_targets)):
            if clargs.index is not None and target_index != clargs.index:
                continue

            targets = all_targets[target_index]

            if clargs.skip_equiv_components and \
               targets.shape[1] != len(find_equivalent_components(targets)):
                continue

            n_distinct_columns = len(set(tuple(targets[:,i]) for i in range(targets.shape[1])))
            print("target index =", target_index)
            print(targets)

            if clargs.randomize_compositions is not None \
               and clargs.scale_component_compositions is not None:
                raise Exception("two methods of composition adjustment have been specified")
            elif clargs.randomize_compositions is not None:
                # Choose each enriched-component composition from a log-uniform distribution:
                targets = np.array(targets, dtype=float)
                for alpha in range(targets.shape[0]):
                    for i in range(targets.shape[1]):
                        targets[alpha,i] += \
                            targets[alpha,i] * math.exp(clargs.randomize_compositions * random.random())
            elif clargs.scale_component_compositions is not None:
                targets = np.array(targets, dtype=float)
                for z in clargs.scale_component_compositions.split(','):
                    i, scalefactor = int(z.split(':')[0]), float(z.split(':')[1])
                    for alpha in range(targets.shape[0]):
                        targets[alpha,i] *= scalefactor

            N = targets.shape[1]
            x = np.zeros(targets.shape)
            for alpha in range(targets.shape[0]):
                x[alpha,:] = targets[alpha,:] / targets[alpha,:].sum()
            L = clargs.L * np.ones(N)

            if clargs.randomize_compositions is not None or \
               clargs.scale_component_compositions is not None:
                print("x =")
                with np.printoptions(formatter={'float': '{: 5.3f}'.format}, linewidth=240):
                    print(x)

            solver, solver_options = solvers[clargs.solver]
            phi_target = clargs.phiTa * x
            try:
                if not 'similarity' in clargs.solver:
                    solution = solver(phi_target, L, clargs.zeta, nonpos=solver_options['nonpos'], \
                                      solver_options={'eps':clargs.solver_tol,\
                                                      'max_iters':clargs.solver_maxiter}, \
                                      **program_options)
                else:
                    solution = solver(phi_target, L, clargs.zeta, eps_ref, \
                                      nonpos=solver_options['nonpos'], \
                                      solver_options={'eps':clargs.solver_tol,\
                                                      'max_iters':clargs.solver_maxiter}, \
                                      **program_options)
            except ProgramException as e:
                print(e)
                print("Skipping...\n")
                continue
            if solution['problem'].status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print("problem status: %s" % solution['problem'].status)
                print("Skipping...\n")
                continue
            eps = solution['eps'].matrix()
            r = np.linalg.matrix_rank(eps)
            if 'similarity' in clargs.solver:
                solution['eps'].set_close_elements_equal(eps_ref)
            mu0 = solution['mu'].vector()
            with np.printoptions(formatter={'float': '{: 5.2f}'.format}, linewidth=240):
                print("eps =")
                print(eps)
                if 'similarity' in clargs.solver:
                    print("(eps - eps_ref) =")
                    print(eps - eps_ref)
                    print(" number of unchanged elements =", \
                          count_close_elements(solution['eps'], eps_ref))
                print("mu =")
                print(mu0)
            optimal_program_data = {'obj' : {k : v.value for k,v in solution['objective'].items()}, \
                                    'status' : solution['problem'].status}
            if 'similarity' in clargs.solver:
                optimal_program_data['count_close_elements'] = \
                    count_close_elements(solution['eps'], eps_ref)
            print(optimal_program_data)

            F = define_free_energy_fh(L, eps)
            phi0 = estimate_dilute_phi(L, mu0)
            phi_est = estimate_phi_targets(phi_target, L, clargs.zeta)
            if clargs.verbose:
                test_verbose = 2
            else:
                test_verbose = 0
            try:
                test_results_dilute = test_landscape_dilute(F, mu0, phi0, verbose=test_verbose)
            except (SolverException, np.linalg.LinAlgError, ValueError):
                test_results_dilute = {'success' : False}
            try:
                test_results_targets = test_landscape_targets(F, mu0, phi_est, verbose=test_verbose)
            except (SolverException, np.linalg.LinAlgError, ValueError):
                test_results_targets = {'success' : False}
            print("LANDSCAPE TESTS:", test_results_dilute['success'], test_results_targets['success'])

            print()

            if clargs.output is not None:
                pickle.dump({'target_index' : target_index, \
                             'targets' : targets, 'phiTa' : clargs.phiTa, \
                             'zeta' : clargs.zeta, 'L' : clargs.L, \
                             'program_options' : program_options, \
                             'eps' : solution['eps'], 'r' : r, \
                             'mu' : solution['mu'], \
                             'optimal_program_data' : optimal_program_data, \
                             'warnings' : [], 'test_results_dilute' : test_results_dilute, \
                             'test_results_targets' : test_results_targets}, \
                            foutput)
                foutput.flush()
