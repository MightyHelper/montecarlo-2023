#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string>
#include <thread>
#include <mutex>
#include <unistd.h>

#define DTYPE long long int
#define N_CONVERGED 10

using namespace std;


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
void apply_metropolis_ising(int matrix[width][height], double temperature,
                            bool periodic_boundary_conditions, double *total_magnetization, double *total_energy) {
	for (int iter = 0; iter < width * height; iter++) {
		int i = rand() % width;
		int j = rand() % height;
		int s, s_up, s_down, s_left, s_right;
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
		double delta_E = ((double) 4.0 * s * (s_up + s_down + s_left + s_right));
		if (delta_E <= 0) {
			matrix[i][j] = -s;
			(*total_magnetization) -= s << 1;
			(*total_energy) += delta_E;
		} else {
			double r = (double) rand() / (double) RAND_MAX;
			if (temperature > 0 and r < exp(-(double) delta_E / temperature)) {
				matrix[i][j] = -s;
				(*total_magnetization) -= s << 1;
				(*total_energy) += delta_E;
			}
		}
	}

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
				printf("X ");
			} else {
				printf("  ");
			}
		}
		printf("\n");
	}
}

template<DTYPE width, DTYPE height>
bool has_converged(int matrix[width][height], double last_energies[N_CONVERGED], int &last_energies_idx,
                   double &last_energies_sum, double &last_energies_avg, double &last_energies_std) {
	double current_energy = get_current_energy<width, height>(matrix);
	last_energies_sum -= last_energies[last_energies_idx];
	last_energies_sum += current_energy;
	last_energies[last_energies_idx] = current_energy;
	last_energies_idx = (last_energies_idx + 1) % N_CONVERGED;
	last_energies_avg = last_energies_sum / N_CONVERGED;
	double sum_of_squares = 0;
	for (int i = 0; i < N_CONVERGED; i++) sum_of_squares += pow(last_energies[i] - last_energies_avg, 2);
	last_energies_std = sqrt(sum_of_squares / N_CONVERGED);
//	printf("Last energies std: %f\n", last_energies_std);
//	printf("Last energies avg: %f\n", last_energies_avg);
//	fflush(stdout);
	return last_energies_std < 1.5 / (double) width;
}

template<DTYPE width, DTYPE height>
void init(int matrix[width][height], double *total_magnetization) {
	*total_magnetization = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			double r = (double) rand() / (double) RAND_MAX;
			if (r < 0.5) {
				matrix[i][j] = 1;
				(*total_magnetization) += 1;
			} else {
				matrix[i][j] = -1;
				(*total_magnetization) -= 1;
			}
		}
	}
}

template<int width, int height, int times>
void run_full_sim(double temperature, bool pbc, FILE *output_file, mutex &m) {
	for (int i = 0; i < times; i++) {
		int matrix[width][height] = {0};
		double *total_magnetization = new double(0.0);
		double *total_energy = new double(0.0);
		init<width, height>(matrix, total_magnetization);
		(*total_energy) = get_current_energy<width, height>(matrix) * (width * height);
		double last_energies[N_CONVERGED] = {0};
		int last_energies_idx = 0;
		double last_energies_sum = 0;
		double last_energies_avg = 0;
		double last_energies_std = 0;
//		printf("Energy: %f / %f\n", (*total_energy) / (width * height), get_current_energy<width, height>(matrix));
//		printf("Magnetization: %f / %f\n", abs((*total_magnetization) / (width * height)), get_current_magnetization<width, height>(matrix));
		for (int i = 0; i < N_CONVERGED; i++) {
			double start_energy = get_current_energy<width, height>(matrix);
			apply_metropolis_ising<width, height>(matrix, temperature, pbc, total_magnetization, total_energy);
			double end_energy = get_current_energy<width, height>(matrix);
			last_energies[i] = (*total_energy);
			last_energies_sum += last_energies[i];
		}
		while (!has_converged<width, height>(
			matrix,
			last_energies,
			last_energies_idx,
			last_energies_sum,
			last_energies_avg,
			last_energies_std
		)) {
			apply_metropolis_ising<width, height>(matrix, temperature, pbc, total_magnetization, total_energy);
		}
		m.lock();
		fprintf(output_file, "%f %f %f %d\n",
		        (*total_energy) / (width * height),
		        abs((*total_magnetization) / (width * height)),
		        temperature,
		        width * height
		);
		m.unlock();
		delete total_magnetization;
		delete total_energy;
	}
}

template<int size>
void run_simulation(float simulation_width = 1, float squishification_degree = 2, float simulation_count = 5) {
	constexpr DTYPE width = size;
	constexpr DTYPE height = size;
	constexpr DTYPE system_size = width * height;
	float p_0 = 2.26918531421;
	// exec_{size}.txt
	auto filename = (string("exec_") + to_string(size) + ".txt").c_str();
	FILE *output_file = fopen(filename, "w");
	bool pbc = true;
	mutex m;
	for (int idx = -simulation_count; idx < simulation_count; idx += 1) {
		double temperature = (
			simulation_width * idx *
			(pow(abs(idx), (squishification_degree - 1))) / (pow(simulation_count, squishification_degree))
			+ p_0
		);
		thread t0(run_full_sim<width, height, 2>, temperature, pbc, output_file, ref(m));
		thread t1(run_full_sim<width, height, 2>, temperature, pbc, output_file, ref(m));
		thread t2(run_full_sim<width, height, 2>, temperature, pbc, output_file, ref(m));
		thread t3(run_full_sim<width, height, 2>, temperature, pbc, output_file, ref(m));
		t0.join();
		t1.join();
		t2.join();
		t3.join();
		printf("Done %f/%f : %d\n", idx + simulation_count, 2 * simulation_count, size);
	}
	fclose(output_file);
}

int main() {
//	srand(time(NULL));
	srand(1);
	float w = 1;
	float degree = 2;
	float count = 100;
	FILE *output_file = fopen("exec.txt", "w");
	mutex m;
//	run_full_sim<32, 32, 1>(1, true, output_file, ref(m));
// Run in thread
	printf("Running in threads\n");
#define sim(thr, sz) thread thr(run_simulation<sz>, w, degree, count)
	sim(ta, 8);
	sim(t0, 16);
	sim(t1, 32);
	sim(t2, 64);
	ta.join();
	t0.join();
	t1.join();
	t2.join();

	return 0;
}
