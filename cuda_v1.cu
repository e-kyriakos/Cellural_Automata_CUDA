#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

// ANSI color escape codes
#define RED "\x1B[31m"
#define GREEN "\x1B[32m"
#define RESET "\x1B[0m"

// Function to initialize the Ising model with a random initial state
void initialize(int *grid, int n) {
    for (int i = 0; i < n * n; ++i) {
        grid[i] = (rand() % 2) * 2 - 1; // +1 or -1
    }
}

// CUDA kernel for simulating one iteration of the Ising model
__global__ void update(int *current, int *next, int n) {
	int i = blockIdx.x;
	int j = blockIdx.y;

        // Calculate the new state based on the majority of neighbors
        int sum = current[((i - 1 + n) % n) * n + j] +
                  current[i * n + (j - 1 + n) % n] +
                  current[i * n + j] +
                  current[((i + 1) % n) * n + j] +
                  current[i * n + (j + 1) % n];

        next[i * n + j] = (sum > 0) ? 1 : -1;
}

// Function to print the current state of the Ising model
void printState(int *grid, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
		if(grid[i * n + j] == 1){
	  		printf(GREEN "■" RESET);
		}
		else{
	  		printf(RED "■" RESET);
        	}
    	}
        printf("\n");
    }
    printf("-----------------------------------------------\n");
}

int main(int argc, char *argv[]) {

    int seed = 42;
    srand(seed);

    if (argc < 2 && argc > 3) {
        printf("Usage: %s <size_n> <iterations_k> <a>\n", argv[0]);
        return 1;
    }

    struct timeval t1, t2;
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    char c;


    if (argc == 4) {
        c = *argv[3];
        printf("c = %c\n", c);
    } else {
        c = '\0';
    }

    gettimeofday(&t1, 0);

    // Allocate memory for two grids (current and next states) on GPU
    int *device_grid1, *device_grid2;
    cudaMalloc((void **)&device_grid1, n * n * sizeof(int));
    cudaMalloc((void **)&device_grid2, n * n * sizeof(int));

    // Initialize the Ising model with a random initial state
    int *host_grid1 = (int *)malloc(n * n * sizeof(int));
    initialize(host_grid1, n);
    cudaMemcpy(device_grid1, host_grid1, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(1);
    dim3 gridDim(n, n);

    // Simulation loop on GPU
    for (int i = 0; i < k; ++i) {
        // Perform one iteration of the Ising model
        update<<<gridDim, blockDim>>>(device_grid1, device_grid2, n);

        // Swap the pointers for the next iteration
        int *temp = device_grid1;
        device_grid1 = device_grid2;
        device_grid2 = temp;

	// Write 'a' argument for animation of 10 fps
	if(c == 'a'){
         	cudaMemcpy(host_grid1, device_grid1, n * n * sizeof(int), cudaMemcpyDeviceToHost);
         	printState(host_grid1, n);
	 	usleep(100000);
	}
    }

    // Print the final state of the Ising model (if needed)
    cudaMemcpy(host_grid1, device_grid1, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    //printState(host_grid1, n);

    // Free allocated memory on GPU
    cudaFree(device_grid1);
    cudaFree(device_grid2);

    // Free allocated memory on CPU
    free(host_grid1);

    //HANDLE_ERROR(cudaThreadSynchronize());
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to generate:  %3.1f ms \n", time);

    return 0;
}
