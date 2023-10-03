#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const int width = 5000;
const int height = width;
float temperature = 1.0;

int matrix[width][height];

void apply_metropolis_ising(){
			for(int i = 0; i < width; i++){
				for(int j = 0; j < height; j++){
						int delta_E = 0;
						int s = matrix[i][j];
						int s_up = matrix[i][(j+1)%height];
						int s_down = matrix[i][(j-1+height)%height];
						int s_left = matrix[(i-1+width)%width][j];
						int s_right = matrix[(i+1)%width][j];
						delta_E = 2*s*(s_up + s_down + s_left + s_right);
						if(delta_E < 0){
								matrix[i][j] = -s;
						}
						else{
								double r = (double)rand() / (double)RAND_MAX;
								if(r < exp(-delta_E/temperature)){
										matrix[i][j] = -s;
								}
						}
				}
		}
}

float get_current_energy(){
		float energy = 0;
		for(int i = 0; i < width; i++){
				for(int j = 0; j < height; j++){
						int s = matrix[i][j];
						int s_up = matrix[i][(j+1)%height];
						int s_down = matrix[i][(j-1+height)%height];
						int s_left = matrix[(i-1+width)%width][j];
						int s_right = matrix[(i+1)%width][j];
						energy += -s*(s_up + s_down + s_left + s_right);
				}
		}
		return energy;
}

float get_current_magnetization(){
		float magnetization = 0;
		for(int i = 0; i < width; i++){
				for(int j = 0; j < height; j++){
						magnetization += matrix[i][j];
				}
		}
		return magnetization;
}

void print_matrix(){
		for(int i = 0; i < width; i++){
				for(int j = 0; j < height; j++){
						if(matrix[i][j] == 1){
								printf("X");
						}
						else{
								printf(" ");
						}
				}
				printf("\n");
		}
}

int main(){
		srand(time(NULL));
		for(int i = 0; i < width; i++){
				for(int j = 0; j < height; j++){
						double r = (double)rand() / (double)RAND_MAX;
						if(r < 0.5){
								matrix[i][j] = 1;
						}
						else{
								matrix[i][j] = -1;
						}
				}
		}
		int system_size = width*height;
		for(int i = 0; i < 1000; i++){
				apply_metropolis_ising();
				if(i % 10 == 0){
						printf("E:%f M:%f i:%d\n", get_current_energy() / system_size, get_current_magnetization() / system_size, i);
				}
		}
		print_matrix();
		return 0;
}
