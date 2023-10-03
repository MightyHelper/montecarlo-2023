import os

import numpy as np
from numba import njit, config
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import time
from multiprocessing import Pool

# config.DISABLE_JIT = True
# config.NUMBA_NUM_THREADS = 1

print("Compiling...", flush=True)


@njit
def apply_metropolis_importance_sampling(state, site, temperature, pbc=(True, True)):
    """Apply the Metropolis importance sampling algorithm to the given state."""
    # Compute the change in energy
    delta_energy = 2 * compute_energy_at_site2d(state, site, pbc)
    # print(delta_energy)
    # current_energy = compute_energy(state, pbc)
    # state[site] *= -1
    # new_energy = compute_energy(state, pbc)
    # delta_energy_n = new_energy - current_energy
    # if (delta_energy_n - delta_energy_o) != 0:
    #     print(">>", delta_energy_n, delta_energy_o, current_energy, new_energy)
    # delta_energy = delta_energy_n
    # If the change in energy is negative, accept the move
    if delta_energy <= 0:
        state[site] *= -1
        return
    # print(f"{delta_energy=}, {temperature=}, {np.exp(-delta_energy / temperature)}")
    if temperature > 0 and np.random.rand() < np.exp(-delta_energy / temperature):
        state[site] *= -1


def apply_metropolis(state, temperature):
    """Apply the Metropolis algorithm to the given state."""
    # Choose a random site
    site = tuple(np.random.randint(0, state.shape[0], state.ndim))
    # Apply the Metropolis importance sampling algorithm
    apply_metropolis_importance_sampling(state, site, temperature)


@njit
def apply_metropolis2d(state, temperature, pbc=(True, True)):
    """Apply the Metropolis algorithm to the given state."""
    # Choose a random site
    site_x = np.random.randint(0, state.shape[0])
    site_y = np.random.randint(0, state.shape[0])
    # Apply the Metropolis importance sampling algorithm
    apply_metropolis_importance_sampling(state, (site_x, site_y), temperature, pbc)


@njit
def print_state(state):
    mag = compute_magnetization(state)
    span = "█" * 2
    white = "\033[37m"
    black = "\033[30m"
    green = "\033[32m"
    red = "\033[31m"
    rst = "\033[0m"
    border = green if mag > 0 else red
    line = border + span * (state.shape[0] + 2) + rst
    rows = [compute_row(black, border, i, rst, span, state, white) for i in range(state.shape[0])]
    print(line + "\n" + "".join(rows) + line)


@njit
def compute_row(black, border, i, rst, span, state, white):
    last = 0
    out = border + span
    for j in range(state.shape[1]):
        if last != state[i, j]:
            last = state[i, j]
            out += rst + (white if last == 1 else black)
        out += span
    out += border + span + rst + "\n"
    return out


@njit
def get_initial_state(sim_box_size):
    state = np.random.randint(0, 2, (sim_box_size, sim_box_size))
    state *= 2
    state -= 1
    return state


@njit
def print_data(state, pbc=(True, True)):
    print("Magnetization:")
    print(compute_magnetization(state))
    print("Energy:")
    print(compute_energy(state, pbc))
    # print(f"State:")
    # print_sate(state)


@njit(parallel=True)
def compute_magnetization(state):
    return np.sum(state) / state.size


def compute_energy_at_site(state, site, pbc=(True, True)):
    energy = 0
    for i in range(len(state.shape)):
        for j in [-1, 1]:
            neighbor = list(site)
            neighbor[i] = (neighbor[i] + j)
            if pbc[i]:
                neighbor[i] %= state.shape[i]
            if neighbor[i] < 0 or neighbor[i] >= state.shape[i]:
                continue
            energy -= state[tuple(neighbor)]
    return energy * state[site]


@njit
def compute_energy_at_site2d(state, site, pbc=(True, True)):
    energy = 0
    n_x, n_y = site
    # out = ""
    for j in [-1, 1]:
        n_xx = n_x + j
        if pbc[0]: n_xx %= state.shape[0]
        if 0 <= n_xx < state.shape[0]:
            energy += state[n_xx, n_y]
        # out += f"{n_xx}, {n_y} [{state[n_xx, n_y]}] | "
    for j in [-1, 1]:
        n_yy = n_y + j
        if pbc[1]: n_yy %= state.shape[1]
        if 0 <= n_yy < state.shape[1]:
            energy += state[n_x, n_yy]
        # out += f"{n_x}, {n_yy} [{state[n_x, n_yy]}] | "
    # print(out + f"= {energy} {state[site]}")
    return energy * state[site]


@njit
def compute_energy(state, pbc=(True, True)):
    return sum([sum([compute_energy_at_site2d(state, (i, j), pbc) for j in range(state.shape[1])]) for i in
                range(state.shape[0])]) / state.size


@njit
def run_simulation(sim_box_size, temperature, n_samples, pbc=(True, True)):
    state = get_initial_state(sim_box_size)
    n_cell = sim_box_size * sim_box_size * 20
    mag_log = []
    ener_log = []
    for i in range(n_samples // n_cell):
        for _ in range(n_cell):
            apply_metropolis2d(state, temperature, pbc)

        # time.sleep(0.1)
        # print_state(state)
        # print_data(state, pbc)
        mag_log.append(compute_magnetization(state))
        ener_log.append(compute_energy(state, pbc))

    return state, mag_log, ener_log


def main():
    print("Running...")
    c = 2.26918531421
    # generate_gif(8, 1.1668462899360006, True, 20000)
    for size in [32, 64, 128, 256, 512, 1024]:
        generate_gif(size, c, True, 10000)
        generate_gif(size, c, False, 10000)
    # do_single_run(32, 3, True, 30000)
    # do_single_run(32, 4, True)
    # do_single_run(32, 1, False)
    # do_single_run(32, 4, False)


def do_single_run(size, temp, pbc, sim_count):
    fname, fname2, centroid, np_mag, np_energs = run_sim_for_temperature(size, temp, pbc, sim_count)
    plt.clf()
    plt.plot(np_energs)
    plt.plot(np_mag)
    plt.legend(["Energy", "Magnetism"])
    plt.title(f"Energy and Magnetism vs Time\nPBC: {pbc}, Size: {size}, Temperature: {temp}")
    plt.xlabel("Time / 100")
    plt.ylabel("Energy and Magnetism")
    fname3 = f"energ_mag_{size}_{pbc}_{temp}.png"
    plt.savefig(fname3)


def generate_gif(size, p_0, pbc=True, sim_count=16000):
    images = []
    imagesa = []
    centroids = []
    # See https://www.desmos.com/calculator/j2rpoekt1n
    simulation_count = 10  # on each side of the transition temperature
    squishification_degree = 1  # higher = more samples near the transition temperature
    simulation_width = 0.2  # width of the simulation (p_0 +- simulation_width)
    get_raw_range = lambda: np.arange(-simulation_count, simulation_count + 0.5, 1)
    get_temp = lambda i: (simulation_width * i * (abs(i) ** (squishification_degree - 1)) /
                          (simulation_count ** squishification_degree) + p_0)
    sim_range = [x for x in get_raw_range() if get_temp(x) >= 0]
    mag_list = []
    ener_list = []
    for fname, fname2, centroid, mags, energs in run_simulations(get_temp, pbc, sim_count, sim_range, size):
        print(">>", fname)
        images.append(fname)
        imagesa.append(fname2)
        centroids.append(centroid)
        mag_list.append(mags)
        ener_list.append(energs)
    print(len(mag_list), len(ener_list), len(centroids))
    df = pd.DataFrame({
        "magnetization": mag_list,
        "energy": ener_list,
        "temperature": sim_range
    })
    df.to_pickle(f"df_{size}_{pbc}_{p_0}.pkl")
    # Images to gif
    imageio.mimsave(f'plots/ising_r_{size}_{pbc}_{p_0}.gif', [imageio.imread(fn) for fn in images], duration=0.1,
                    loop=0)
    imageio.mimsave(f'plots/ising_a_{size}_{pbc}_{p_0}.gif', [imageio.imread(fn) for fn in imagesa], duration=0.1,
                    loop=0)
    plot_avg_magnetism(size, centroids, get_temp, p_0, pbc, sim_range)
    # remove pngs
    for fname in images + imagesa:
        os.remove(fname)


def run_simulations(get_temp, pbc, sim_count, sim_range, size):
    if config.DISABLE_JIT:
        with Pool() as p:
            return p.starmap(run_sim_for_temperature, [(size, get_temp(i), pbc, sim_count) for i in sim_range])
    else:
        return [run_sim_for_temperature(size, get_temp(i), pbc, sim_count) for i in sim_range]



def plot_avg_magnetism(size, centroids, get_temp, p_0, pbc, sim_range):
    plt.clf()
    x = [get_temp(i) for i in sim_range]
    centroids = np.array(centroids)
    n_smooth = 2 * int(len(centroids) ** 0.5)
    print(f"{n_smooth=}")
    smooth_centroids = np.convolve(centroids, np.ones(n_smooth) / n_smooth, mode='valid')
    smooth_centroids_x_coords = x[n_smooth // 2: -n_smooth // 2 + 1]
    derivative_of_smooth_centroids = np.gradient(smooth_centroids, smooth_centroids_x_coords)
    argmin = np.argmin(derivative_of_smooth_centroids)
    max_derivative_temp = x[argmin + n_smooth // 2]
    print(
        f"{max_derivative_temp=}, {argmin=}, {x[argmin]=}, {smooth_centroids=}, {smooth_centroids_x_coords=}, {derivative_of_smooth_centroids=}")
    print(f"{max_derivative_temp=} {argmin=}")
    plt.plot(x, centroids)
    # add an x at each data point
    plt.scatter(x, centroids, c='r', marker='x', alpha=0.1)
    plt.plot(smooth_centroids_x_coords, smooth_centroids)
    plt.axvline(x=max_derivative_temp, color='r', linestyle='--')
    plt.legend(["Magnetism", "Magnetism data points", "Smoothed Magnetism", "Min Derivative"])
    plt.xlabel("Temperature")
    plt.ylabel("Average Magnetism")
    plt.ylim(0, 1)
    plt.title(f"Avg Magnetism vs Temperature\nPBC: {pbc}, Size: {size}\nTransition temperature: {max_derivative_temp}")
    fname = f"plots/centroid_{size}_{pbc}_{p_0}.png"
    fname_g = f"plots/centroid_gradient_{size}_{pbc}_{p_0}.png"
    plt.savefig(fname)
    plt.clf()
    plt.plot(smooth_centroids_x_coords, derivative_of_smooth_centroids)
    plt.axvline(x=max_derivative_temp, color='r', linestyle='--')
    plt.legend(["Derivative of Magnetism", "Min Derivative"])
    plt.xlabel("Temperature")
    plt.ylabel("Derivative of Average Magnetism")
    plt.title(
        f"Derivative of Avg Magnetism vs Temperature\nPBC: {pbc}, Size: {size}\nTransition temperature: {max_derivative_temp}")
    plt.savefig(fname_g)


def run_sim_for_temperature(size, temp, pbc, sim_count):
    state, mag, energs = run_simulation(size, temp, size * size * sim_count, (pbc, pbc))
    np_mag = np.array(mag)
    np_energs = np.array(energs)
    plt.clf()
    plt.scatter(np_mag, np_energs, c=np.arange(len(np_mag)), cmap='viridis', alpha=0.1)
    plt.xlabel("Magnetization")
    plt.ylabel("Energy")
    plt.xlim(-1, 1)
    plt.ylim(-8, 8)
    plt.title(f"PBC: {pbc}, Temperature: {temp}")
    fname = f"plots/mag_r_energ_{size}_{pbc}_{temp}.png"
    plt.savefig(fname)
    plt.clf()
    plt.scatter(np.abs(np_mag), np_energs, c=np.arange(len(np_mag)), cmap='viridis', alpha=0.1)
    plt.xlabel("Absolute Magnetization")
    plt.ylabel("Energy")
    plt.xlim(0, 1)
    plt.ylim(-8, 8)
    plt.title(f"PBC: {pbc}, Temperature: {temp}")
    fname2 = f"plots/mag_a_energ_{size}_{pbc}_{temp}.png"
    plt.savefig(fname2)
    centroid = np.mean(np.abs(np_mag[-(len(np_mag) // 8)]))
    print(fname)
    return fname, fname2, centroid, np_mag, np_energs


# def formula_for_E(temperature):
#     # U = - J \coth(2 \beta J) \left[ 1 + \frac{2}{\pi} (2 \tanh^2(2 \beta J) -1) \int_0^{\pi/2} \frac{1}{\sqrt{1 - 4 k (1+k)^{-2} \sin^2(\theta)}} d\theta \right]
#     j = 1
#     beta = 1
#     beta2j = 2 * beta * j
#     k = 1 / (temperature * beta)
#     integral_result = scipy.integrate.quad(
#         lambda theta: 1 / np.sqrt(1 - 4 * k * (1 + k) ** (-2) * np.sin(theta) ** 2),
#         0,
#         np.pi / 2
#     )[0]
#     coth = 1 / np.tanh(beta2j)
#     u = (-j * coth * (1 + 2 / np.pi * (2 * np.tanh(beta2j) ** 2 - 1) * integral_result))
#
#     return u
#
#
# def get_critical_temperature():
#     # {\displaystyle {\frac {kT_{c}}{J}}={\frac {2}{\ln(1+{\sqrt {2}})}}\approx 2.26918531421}
#     k = 1
#     j = 1
#     return 2 * j / np.log(1 + np.sqrt(2))
#
# def formula_for_M():
#     # M = \left[ 1 - \sinh^{-4}(2 \beta J) \right]^{1/8}
#     j = 1
#     beta = 1
#     m = (1 - np.sinh(2 * beta * j) ** (-4)) ** (1 / 8)
#     return m
#

if __name__ == "__main__":
    # print(formula_for_E(1.0))
    # print(formula_for_M())
    main()
