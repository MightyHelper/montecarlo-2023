#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DTYPE long long int

const DTYPE width = 32;
const DTYPE height = width;
const DTYPE system_size = width * height;
double temperature = 1;
bool periodic_boundary_conditions = true;

int matrix[width][height];

void apply_metropolis_ising() {
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

double get_current_energy() {
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
	return energy / system_size;
}

double compute_energy_at(int i, int j) {
	double energy = 0;
	int s = matrix[i][j];
	int s_up = matrix[i][(j + 1) % height];
	int s_down = matrix[i][(j - 1 + height) % height];
	int s_left = matrix[(i - 1 + width) % width][j];
	int s_right = matrix[(i + 1) % width][j];
	energy += -s * (s_up + s_down + s_left + s_right);
	return energy;
}

double get_current_magnetization() {
	double magnetization = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			magnetization += matrix[i][j];
		}
	}
	return magnetization / system_size;
}

void print_matrix() {
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

bool has_converged() {
	static double last_energies[10] = {0};
	static int last_energies_idx = 0;
	static double last_energies_sum = 0;
	static double last_energies_avg = 0;
	static double last_energies_std = 0;
	double current_energy = get_current_energy();
	last_energies_sum -= last_energies[last_energies_idx];
	last_energies_sum += current_energy;
	last_energies[last_energies_idx] = current_energy;
	last_energies_idx = (last_energies_idx + 1) % 10;
	last_energies_avg = last_energies_sum / 10;
	double sum_of_squares = 0;
	for (int i = 0; i < 10; i++) {
		sum_of_squares += pow(last_energies[i] - last_energies_avg, 2);
	}
	last_energies_std = sqrt(sum_of_squares / 10);
//	printf("E:%f E_avg:%f E_std:%f\n", current_energy, last_energies_avg, last_energies_std);
	return last_energies_std < 0.1 * width * height * 0.01;
}

void init() {
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

int main() {
//	srand(time(NULL));
	srand(1);
	for (temperature = 0; temperature < 3; temperature+=0.01) {
		for (int run = 0; run < 100; run++) {
			init();
			int i = 0;
			do {
				apply_metropolis_ising();
				if (i % 10 == 0) {
//				printf("E:%f M:%f i:%d\n", get_current_energy(), get_current_magnetization(), i + 1);
				}
				i++;
			} while (!has_converged() or i < 10);
			printf("E: %f M: %f iter: %lld temp: %f\n", get_current_energy(), abs(get_current_magnetization()),
			       i * system_size, temperature);
		}
	}
	return 0;
}
