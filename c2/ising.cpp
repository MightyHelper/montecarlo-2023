#include <random>
#include <mpi.h>
#include <omp.h>
std::random_device rd;

template<long int size>
long long int mat_or0(int matrix[size][size], int i, int j) {
	if (i < 0 or i >= size or j < 0 or j >= size) return 0;
	else return matrix[i][j];
}

template<long int size>
long long int get_neighbour_sum(int matrix[size][size], int i, int j, bool periodic_x, bool periodic_y) {
	long long int sum = 0;
	sum += mat_or0(matrix, periodic_x ? (i - 1 + size) % size : i - 1, j);
	sum += mat_or0(matrix, periodic_x ? (i + 1) % size : i + 1, j);
	sum += mat_or0(matrix, i, periodic_y ? (j - 1 + size) % size : j - 1);
	sum += mat_or0(matrix, i, periodic_y ? (j + 1) % size : j + 1);
	return -matrix[i][j] * sum;
}


template<long int size>
long long calculate_Energy(int matrix[size][size], bool periodic_x, bool periodic_y) {
	long long energy = 0;
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			energy += get_neighbour_sum(matrix, i, j, periodic_x, periodic_y);
	return energy;
}

template<long int size>
int calculate_Magnetization(int matrix[size][size]) {
	int magnetization = 0;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			magnetization += matrix[i][j];
		}
	}
	return magnetization;
}


template<long int size>
void apply_metropolis(int matrix[size][size], float temperature, bool periodic_x, bool periodic_y) {
	// Initialize random number generator
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> real(0, 1);

	std::uniform_int_distribution<> dis(0, size - 1);
	// Choose random position
	int i = dis(gen);
	int j = dis(gen);
	// Calculate energy difference
	long long int delta_E = -2 * get_neighbour_sum(matrix, i, j, periodic_x, periodic_y);
	// Accept or reject


	if (delta_E <= 0 or (temperature > 0 and real(gen) < exp(-(double) delta_E / temperature))) {
		matrix[i][j] *= -1;
	}
}

template<long int size>
void initialize_matrix(int matrix[size][size], float temperature) {
	// Initialize matrix with random values
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			matrix[i][j] = (dis(gen) < temperature) ? 1 : -1;
		}
	}
}

template<long int size>
void print_matrix(int matrix[size][size]) {
	// Print matrix
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf(matrix[i][j] == 1 ? "XX" : "  ");
		}
		printf("\n");
	}
}

template<long int size>
long long simulate_metropolis_for_magnetization(double temperature) {
	// Initialize matrix
	int matrix[size][size];
	initialize_matrix(matrix, 0.5);
	// Apply metropolis
	for (int i = 0; i < 1000 * size * size; i++){
		apply_metropolis(matrix, temperature, true, true);
		if (i%size * size == 0) printf("%d,%ld,%f,%d,%lld\n", i, size, temperature, abs(calculate_Magnetization(matrix)), calculate_Energy(matrix, true, true));
	}
	// Calculate magnetization
	return abs(calculate_Magnetization(matrix));
}

template<long int size>
long long simulate_metropolis_for_magnetization(double temperature, const int n) {
	long long int magnetization = 0;
#pragma omp parallel for reduction(+:magnetization)
	for (int i = 0; i < n; i++) {
		magnetization += simulate_metropolis_for_magnetization<size>(temperature);
	}
	return magnetization / n;
}


double get_temperature(double simulation_width, int simulation_count, int idx) {
	double squishification_degree = 2;
	idx -= int((double) simulation_count * (double) 0.7); // Offset
	double p_0 = 2.26918531421;
	return (
		simulation_width * idx *
		(pow(abs(idx), (squishification_degree - 1))) / (pow(simulation_count, squishification_degree))
		+ p_0
	);

}

int main() {
	// Initialize matrix
	MPI_Init(nullptr, nullptr);
	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	if (mpi_rank == 0) printf("Running!\n\n");
//	double siumulation_wdith = 3.5;
//	int n_temperatures = 100;
//	int n_temperatures_per_rank = n_temperatures / mpi_size;
	double mag;
#define sim_temp(x, t) mag = simulate_metropolis_for_magnetization<x>(t, 100);
	sim_temp(4, 1);
	sim_temp(8, 1);
	sim_temp(16, 1);
	sim_temp(32, 1);
	sim_temp(64, 1);
	sim_temp(4, 4);
	sim_temp(8, 4);
	sim_temp(16, 4);
	sim_temp(32, 4);
	sim_temp(64, 4);
//	 mag = simulate_metropolis_for_magnetization<size>(4, 1);
//	 printf("T: 4, mag:%f\n", mag / (size * size));
// for (int i = mpi_rank * n_temperatures_per_rank; i < (mpi_rank + 1) * n_temperatures_per_rank; i++) {
// double temperature = get_temperature(siumulation_wdith, n_temperatures, i);
// double magnetization = (double) simulate_metropolis_for_magnetization<size>(temperature, 100) / (double) (size * size);
// printf("%f %f\n", temperature, magnetization);
//		printf("%f\n", temperature);
// }
	MPI_Finalize();
	return 0;
}
