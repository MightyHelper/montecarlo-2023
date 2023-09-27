import random
import multiprocessing
from numba import jit

@jit(nopython=True, inline='always')
def test():
    return random.random()**2 + random.random()**2 < 1

@jit(nopython=True)
def get_pi(terms):
    return sum([test() for _ in range(terms)])

@jit(nopython=True)
def get_pi2(terms):
    total = 0
    for _ in range(terms):
        total += test()
    return total

def get_pi3(terms):
    with multiprocessing.Pool() as pool:
        results = pool.imap_unordered(get_pi2, [terms/32] * 32)
        return sum(results)
runs = 30000000000

# total = sum([test() for _ in range(runs)])
print(4 * get_pi3(runs) / runs)