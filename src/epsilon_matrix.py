import math, random
import numpy as np
import scipy.sparse
import cvxpy as cp

def krdel(i,k):
    if i == k: return 1.
    else: return 0.

# Routines for manipulating symmetric matrices:

def index_matrix(N):
    index = -1 * np.ones((N, N), dtype=int)
    k = 0
    for i in range(index.shape[0]):
        for j in range(i, index.shape[1]):
            index[i,j] = index[j,i] = k
            k += 1
    return index

def get_symm_vector(v, index):
    vk = np.zeros(index.max() + 1)
    for i in range(v.shape[0]):
        for j in range(v.shape[0]):
            if index[i,j] >= 0:
                vk[index[i,j]] = v[i,j]
    return vk

def get_symm_matrix(vk, index):
    v = np.zeros(index.shape)
    for i in range(v.shape[0]):
        for j in range(v.shape[0]):
            if index[i,j] >= 0:
                v[i,j] = vk[index[i,j]]
    return v

class EpsilonMatrix(object):
    def __init__(self, targets, nonpos=False):
        self.targets = targets
        self.nonpos = nonpos
        self.n = self.targets.shape[0]
        self.N = self.targets.shape[1]
        self.index = index_matrix(self.N)
        self.neps = int(self.index.max() + 1)
        self.vec = cp.Variable(self.neps, nonpos=nonpos)
        V2Mlil = scipy.sparse.lil_matrix((self.N*self.N, self.neps))
        for i in range(self.N):
            for j in range(self.N):
                if self.index[i,j] >= 0:
                    V2Mlil[self.N*i+j,self.index[i,j]] = 1.
        self.V2M = scipy.sparse.csr_matrix(V2Mlil)
        self.VW = np.array([1. if i == j else 2. for i in range(self.N) for j in range(i, self.N) \
                            if self.index[i,j] >= 0])
        self.VWh = np.sqrt(self.VW)
        self.Vt = [np.array([1. if self.targets[a,i] > 0. and self.targets[a,j] > 0. else 0. \
                             for i in range(self.N) for j in range(i, self.N) \
                             if self.index[i,j] >= 0]) for a in range(self.n)]
        self.VWa = [self.Vt[a] * self.VW for a in range(self.n)]
        self.VWah = [self.Vt[a] * self.VWh for a in range(self.n)]
    def copy(self):
        copy = EpsilonMatrix(self.targets, nonpos=self.nonpos)
        copy.vec.value = self.vec.value
        return copy
    def matrix(self):
        return get_symm_matrix(self.vec.value, self.index)
    def cp_matrix(self):
        return cp.reshape(self.V2M @ self.vec, (self.N,self.N))
    def cp_min(self):
        return cp.min(self.vec)
    def cp_max(self):
        return cp.max(self.vec)
    def cp_mean(self):
        return (self.VW @ self.vec) / self.N**2
    def cp_variance(self):
        return cp.sum_squares(cp.multiply(self.VWh, self.vec - self.cp_mean())) / self.N**2
    def cp_target_means(self):
        return [(self.VWa[i] @ self.vec) / self.VWa[i].sum() for i in range(self.n)]
    def cp_target_variances(self):
        return [cp.sum_squares(cp.multiply(self.VWah[i], \
                                           self.vec - (self.VWa[i] @ self.vec) / self.VWa[i].sum())) \
                / self.VWa[i].sum() for i in range(self.n)]
    def cp_target_diag(self):
        return [cp.hstack([self.vec[self.index[i,i]] for i in range(self.N) \
                           if self.targets[a,i] > 0.]) for a in range(self.n)]
    def cp_msd(self, mu):
        return cp.sum_squares(cp.multiply(self.VWh, self.vec - mu)) / self.N**2
    def cp_target_msds(self, mus):
        return [cp.sum_squares(cp.multiply(self.VWah[i], self.vec - mus[i])) / self.VWa[i].sum() \
                for i in range(self.n)]

    def cp_nuc_norm(self):
        return cp.norm(self.cp_matrix(), 'nuc')

    def min(self):
        return self.vec.value.min()
    def max(self):
        return self.vec.value.max()
    def mean(self):
        return self.cp_mean().value
    def variance(self):
        return self.cp_variance().value
    def target_means(self):
        return [x.value for x in self.cp_target_means()]
    def target_variances(self):
        return [x.value for x in self.cp_target_variances()]
    def msd(self, mu):
        return self.cp_msd(mu).value
    def target_msds(self, mus):
        return [x.value for x in self.cp_target_msds(mus)]

    def singvals(self):
        return np.array(sorted(math.fabs(x) for x in np.linalg.eigvalsh(self.matrix())))
    def nuc_norm(self):
        return np.linalg.norm(self.matrix(), 'nuc')

    def set_close_elements_equal(self, eps_ref, tol=1.e-3):
        v_ref = get_symm_vector(eps_ref, self.index)
        for i in range(self.neps):
            if math.fabs(self.vec.value[i] - v_ref[i]) <= tol:
                self.vec.value[i] = v_ref[i]

    def print_stats(self, prefix=""):
        print(prefix + "min, max =", self.min(), self.max())
        print(prefix + "mean, var, stddev =", self.mean(), self.variance(), math.sqrt(self.variance()))
        print(prefix + "target means =", self.target_means())
        print(prefix + "target variances =", self.target_variances())
        print(prefix + "eigenvalues =", sorted(np.linalg.eigvals(self.matrix())))

    def low_rank_approx(self, r):
        S, V = np.linalg.eigh(self.matrix())
        order = sorted(range(S.shape[0]), key=lambda i: -math.fabs(S[i]))
        Slr = np.diag([S[i] for i in order[:r]])
        Vlr = np.array([V[:,i] for i in order[:r]]).T
        epslr = np.dot(Vlr, np.dot(Slr, Vlr.T))
        return Slr, Vlr, epslr

class MuVector(object):
    def __init__(self, targets):
        self.N = targets.shape[1]
        self.vec = cp.Variable(self.N)
        self.V2M = np.zeros((self.N*self.N, self.N))
        for i in range(self.N):
            self.V2M[i*self.N:(i+1)*self.N,i] = 1.
    def cp_mean(self):
        return cp.sum(self.vec) / self.N
    def cp_row_diff_matrix(self):
        return cp.reshape(self.V2M @ (self.vec - self.cp_mean()), (self.N, self.N))

    def vector(self):
        return self.vec.value
    def mean(self):
        return self.cp_mean().value
    def row_diff_matrix(self):
        return self.cp_row_diff_matrix().value

def gen_symm_gaussian_noise(eps, sigma):
    v = np.array([random.gauss(0., sigma) for i in range(eps.neps)])
    return get_symm_matrix(v, eps.index)

def count_close_elements(eps, eps_ref, tol=1.e-3):
    v_ref = get_symm_vector(eps_ref, eps.index)
    return sum(1 for i in range(eps.neps) if math.fabs(eps.vec.value[i] - v_ref[i]) <= tol)
