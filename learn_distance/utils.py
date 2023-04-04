import numpy as np

def hinge(c):
    return np.where(c < 0, 0, c)

def log_likelihood(a1, a2, x, d):
    def C(x):
        return 1/(a1(x) + a2(x))
    if len(x.shape) > 1:
        p = -np.sum(a1(x) * hinge(d) + a2(x) * hinge(-d), axis = 1)
        return p
    p = -(a1(x) * hinge(d) + a2(x) * hinge(-d)) #+ np.log(C(x)) need to figure out normalizing constants
    return p

def generate(a1, a2, size=100, mean=0, sd=1, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=mean, scale=sd, size=size)
    p = a1(x) / (a1(x) + a2(x))
    flips = rng.binomial(n=1, p=p)
    d = np.zeros((size))
    d[flips==1] = rng.exponential(a1(x))[flips==1]
    d[flips==0] = -rng.exponential(a2(x))[flips==0]
    return x, d

def generate_asy(a1, a2, size=100, mean=0, sd=1, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=mean, scale=sd, size=size)
    p = a1(x) / (a1(x) + a2(x))
    flips = rng.binomial(n=1, p=p)
    d = np.zeros((size))
    d[flips==1] = rng.exponential(p)[flips==1]
    d[flips==0] = -rng.exponential(1/p)[flips==0]
    return x, d

def generate_d(a1, a2, dim, size=100, basis=20, mean=0, sd=1, seed=123):
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=mean, scale=sd, size=(basis, dim))
    M = rng.uniform(size = (size, basis))
    x = M @ base
    p = a1(x) / (a1(x) + a2(x))
    flips = rng.binomial(n=1, p=p)
    d = np.zeros((size, dim))
    d[np.where(flips==1)] = rng.exponential(a1(x))[np.where(flips==1)]
    d[np.where(flips==0)] = -rng.exponential(a2(x))[np.where(flips==0)]
    return x, d

def generate_d_asy(a1, a2, dim, size=100, basis=20, mean=0, sd=1, seed=123):
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=mean, scale=sd, size=(basis, dim))
    M = rng.uniform(size = (size, basis))
    x = M @ base
    p = a1(x) / (a1(x) + a2(x))
    flips = rng.binomial(n=1, p=p)
    d = np.zeros((size, dim))
    d[np.where(flips==1)] = rng.exponential(p)[np.where(flips==1)]
    d[np.where(flips==0)] = -rng.exponential(1-p)[np.where(flips==0)]
    return x, d