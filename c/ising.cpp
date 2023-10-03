#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string>
#include <thread>

#define DTYPE long long int
#define N_CONVERGED 10

using namespace std;
;

template<DTYPE width, DTYPE height>
void apply_metropolis_ising(int matrix[width][height], double temperature, bool periodic_boundary_conditions) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			int s, s_up, s_down, s_left, s_right, delta_E;
			s = matrix[i][j];
			if (periodic_boundary_conditions) {
				s_up = matrix[i][(j + 1 + height) % height];
				s_down = matrix[i][(j - 1 + height) % height];
				s_left = matrix[(i - 1 + width) % width][j];
				s_right = matrix[(i + 1 + width) % width][j];
			} else {
				s_up = (j == height - 1) ? 0 : matrix[i][j + 1];
				s_down = (j == 0) ? 0 : matrix[i][j - 1];
				s_left = (i == 0) ? 0 : matrix[i - 1][j];
				s_right = (i == width - 1) ? 0 : matrix[i + 1][j];
			}
			delta_E = 2 * s * (s_up + s_down + s_left + s_right);
			if (delta_E <= 0) {
				matrix[i][j] = -s;
			} else {
				double r = (double) rand() / (double) RAND_MAX;
				if (temperature > 0 and r < exp(-(double) delta_E / temperature)) {
					matrix[i][j] = -s;
				}
			}
		}
	}
}
template<DTYPE width, DTYPE height>
double get_current_energy(int matrix[width][height]) {
	double energy = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			int s = matrix[i][j];
			int s_up = matrix[i][(j + 1) % height];
			int s_down = matrix[i][(j - 1 + height) % height];
			int s_left = matrix[(i - 1 + width) % width][j];
			int s_right = matrix[(i + 1) % width][j];
			energy += -s * (s_up + s_down + s_left + s_right);
		}
	}
	return energy / (width * height);
}
template<DTYPE width, DTYPE height>
double compute_energy_at(int i, int j, int matrix[width][height]) {
	double energy = 0;
	int s = matrix[i][j];
	int s_up = matrix[i][(j + 1) % height];
	int s_down = matrix[i][(j - 1 + height) % height];
	int s_left = matrix[(i - 1 + width) % width][j];
	int s_right = matrix[(i + 1) % width][j];
	energy += -s * (s_up + s_down + s_left + s_right);
	return energy;
}
template<DTYPE width, DTYPE height>
double get_current_magnetization(int matrix[width][height]) {
	double magnetization = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			magnetization += matrix[i][j];
		}
	}
	return magnetization / (width * height);
}
template<DTYPE width, DTYPE height>
void print_matrix(int matrix[width][height]) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			if (matrix[i][j] == 1) {
				printf("X");
			} else {
				printf(" ");
			}
		}
		printf("\n");
	}
}
template<DTYPE width, DTYPE height>
bool has_converged(int matrix[width][height], double last_energies[N_CONVERGED], int &last_energies_idx, double &last_energies_sum, double &last_energies_avg, double &last_energies_std) {
	double current_energy = get_current_energy<width, height>(matrix);
	last_energies_sum -= last_energies[last_energies_idx];
	last_energies_sum += current_energy;
	last_energies[last_energies_idx] = current_energy;
	last_energies_idx = (last_energies_idx + 1) % N_CONVERGED;
	last_energies_avg = last_energies_sum / N_CONVERGED;
	double sum_of_squares = 0;
	for (int i = 0; i < N_CONVERGED; i++) sum_of_squares += pow(last_energies[i] - last_energies_avg, 2);
	last_energies_std = sqrt(sum_of_squares / N_CONVERGED);
	return last_energies_std < 0.05;
}
template<DTYPE width, DTYPE height>
void init(int matrix[width][height]) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			double r = (double) rand() / (double) RAND_MAX;
			if (r < 0.6) {
				matrix[i][j] = 1;
			} else {
				matrix[i][j] = -1;
			}
		}
	}
}

template <int size>
void run_simulation(float simulation_width = 1, float squishification_degree = 2, float simulation_count = 5) {
	constexpr DTYPE width = size;
	constexpr DTYPE height = size;
	constexpr DTYPE system_size = width * height;
	int matrix[width][height] = {0};
	float p_0 = 2.26918531421;
	// exec_{size}.txt
	auto filename = (string("exec_") + to_string(size) + ".txt").c_str();
	auto output_file = fopen(filename, "w");
	bool pbc = true;
	for (int idx = -simulation_count; idx < simulation_count; idx += 1) {
		double temperature = (
			simulation_width * idx *
			(pow(abs(idx), (squishification_degree - 1))) / (pow(simulation_count, squishification_degree))
			+ p_0
		);
		for (int run = 0; run < 100; run++) {
			init<width, height>(matrix);
			static double last_energies[10] = {0};
			static int    last_energies_idx = 0;
			static double last_energies_sum = 0;
			static double last_energies_avg = 0;
			static double last_energies_std = 0;
			for (int i = 0; i < 10; i++) apply_metropolis_ising<width, height>(matrix, temperature, pbc);
			while (!has_converged<width, height>(matrix, last_energies, last_energies_idx, last_energies_sum, last_energies_avg, last_energies_std)) apply_metropolis_ising<width, height>(matrix, temperature, pbc);
			fprintf(output_file, "%f %f %f %lld\n",
				get_current_energy<width, height>(matrix),
				abs(get_current_magnetization<width, height>(matrix)),
				temperature,
				width * height
			);
		}
		printf("Done %f/%f : %d\n", idx + simulation_count, 2 * simulation_count, size);
	}
	fclose(output_file);
}

int main() {
//	srand(time(NULL));
	srand(1);
	auto w = 0.5;
	auto degree = 2;
	auto count = 20;
 //run_simulation<300>(w, degree, count);
// Run in thread
#define sim(thr, sz) thread thr(run_simulation<sz>, w, degree, count)
	sim(t0, 16);
	sim(t1, 32);
	sim(t2, 64);
	sim(t3, 128);
	sim(t4, 256);

	t0.join();
	t1.join();
	t2.join();
	t3.join();
	t4.join();

	return 0;
}
