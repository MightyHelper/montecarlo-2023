import numpy as np
from numba import njit
import imageio
import scipy
import matplotlib.pyplot as plt

print("Compiling...", flush=True)


@njit
def compute_energy_change2(state, site, pbc=(True, True)):
    return compute_energy_at_site2d(state, site, pbc)


@njit
def apply_metropolis_importance_sampling(state, site, temperature, pbc=(True, True)):
    """Apply the Metropolis importance sampling algorithm to the given state."""
    # Compute the change in energy
    delta_energy = -compute_energy_change2(state, site, pbc)
    # If the change in energy is negative, accept the move
    if delta_energy <= 0 or (temperature > 0 and np.random.rand() < np.exp(-delta_energy / temperature)):
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
    for j in [-1, 1]:
        n_x, n_y = site
        n_x = n_x + j
        if pbc[0]: n_x %= state.shape[0]
        if 0 <= n_x < state.shape[0]:
            energy -= state[n_x, n_y]
    for j in [-1, 1]:
        n_x, n_y = site
        n_y = n_y + j
        if pbc[1]: n_y %= state.shape[1]
        if 0 <= n_y < state.shape[1]:
            energy -= state[n_x, n_y]
    return energy * state[site]


@njit
def compute_energy(state, pbc=(True, True)):
    return sum([sum([compute_energy_at_site2d(state, (i, j), pbc) for j in range(state.shape[1])]) for i in
                range(state.shape[0])]) / state.size


@njit
def run_simulation(sim_box_size, temperature, n_samples, pbc=(True, True)):
    state = get_initial_state(sim_box_size)
    n_cell = sim_box_size * sim_box_size  # 100
    mag_log = []
    ener_log = []
    for i in range(n_samples // n_cell):
        for _ in range(n_cell):
            apply_metropolis2d(state, temperature, pbc)
        mag_log.append(compute_magnetization(state))
        ener_log.append(compute_energy(state, pbc))

    return state, mag_log, ener_log


def main():
    print("Running...")
    c = 2.26918531421
    generate_gif(1.14, 0.001, True)
    generate_gif(1.14, 0.001, False)


def generate_gif(p_0, z, pbc=True):
    images = []
    imagesa = []
    centroids = []
    get_temp = lambda i: z * (i ** 3) + p_0
    for i in np.arange(-10, 10, 1):
        temp = get_temp(i)
        fname, centroid = run_sim_for_temperature(temp, pbc)
        images.append(fname)
        imagesa.append("a" + fname)
        centroids.append(centroid)
        print(fname)

        # print("-----")
    # Images to gif
    imageio.mimsave(f'ising_{pbc}_{p_0}_{z}.gif', [imageio.imread(fn) for fn in images], duration=1, loop=0)
    imageio.mimsave(f'aising_{pbc}_{p_0}_{z}.gif', [imageio.imread(fn) for fn in imagesa], duration=1, loop=0)
    # Centroids gif
    plt.clf()
    plt.plot([get_temp(i) for i in np.arange(-10, 10, 1)], centroids)
    plt.xlabel("Temperature")
    plt.ylabel("Centroid")
    plt.xlim(get_temp(-10), get_temp(10))
    plt.ylim(0, 1)
    plt.title(f"Centroid vs Temperature, PBC: {pbc}")
    fname = f"centroid_{pbc}_{p_0}_{z}.png"
    plt.savefig(fname)


def run_sim_for_temperature(temp, pbc=True):
    state, mag, energs = run_simulation(16, temp, 1_000_000, (pbc, pbc))
    np_mag = np.array(mag)
    np_energs = np.array(energs)
    # First derivative
    # mag_smooth = np.convolve(np_mag, np.ones(100) / 100, mode='valid')
    # mag_smooth_smooth = np.convolve(mag_smooth, np.ones(100) / 100, mode='valid')
    # mag_smooth_smooth_diff = np.diff(mag_smooth_smooth)
    # energ_smooth = np.convolve(np_energs, np.ones(100) / 100, mode='valid')
    # energ_smooth_smooth = np.convolve(energ_smooth, np.ones(100) / 100, mode='valid')
    # energ_smooth_smooth_diff = np.diff(energ_smooth_smooth)
    # Plot

    # Set color to index
    plt.clf()
    plt.scatter(np_mag, np_energs, c=np.arange(len(np_mag)), cmap='viridis', alpha=0.1)
    plt.xlabel("Magnetization")
    plt.ylabel("Energy")
    plt.xlim(-1, 1)
    plt.ylim(-4.5, 0.5)
    plt.title(f"Temperature: {temp}, PBC: {pbc}")
    fname = f"mag_energ_{pbc}_{temp}.png"
    plt.savefig(fname)
    plt.clf()
    plt.scatter(np.abs(np_mag), np_energs, c=np.arange(len(np_mag)), cmap='viridis', alpha=0.1)
    plt.xlabel("Absolute Magnetization")
    plt.ylabel("Energy")
    plt.xlim(0, 1)
    plt.ylim(-4.5, 0.5)
    plt.title(f"Temperature: {temp}, PBC: {pbc}")
    fname2 = f"amag_energ_{pbc}_{temp}.png"
    plt.savefig(fname2)

    centroid = np.mean(np.abs(np_mag))
    return fname, centroid


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
