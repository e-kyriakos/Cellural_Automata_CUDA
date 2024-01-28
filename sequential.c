#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ANSI color escape codes
#define RED "\x1B[31m"
#define GREEN "\x1B[32m"
#define RESET "\x1B[0m"

// Function to initialize the Ising model with a random initial state
void initialize(int *grid, int n) {
    for (int i = 0; i < n * n; ++i) {
        grid[i] = (rand() % 2) * 2 - 1; // +1 or -1 homogenious distribution
    }
}

// Function to simulate one iteration of the Ising model
void update(int *current, int *next, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // Calculate the new state based on the majority of neighbors
 	int sum = current[((i - 1 + n) % n) * n + j] + current[i * n + (j - 1 + n) % n] + current[i * n + j] + current[((i + 1) % n) * n + j] + current[i * n + (j + 1) % n];
        next[i * n + j] = (sum > 0) ? 1 : -1;
        }
    }
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

    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    char c;

    if (argc == 4) {
        c = *argv[3];
        printf("c = %c\n", c);
    } else {
        c = '\0';
    }

    clock_t cpu_start, cpu_end;
    cpu_start = clock();

    // Allocate memory for two grids (current and next states)
    int *grid1 = (int *)malloc(n * n * sizeof(int));
    int *grid2 = (int *)malloc(n * n * sizeof(int));

    // Initialize the Ising model with a random initial state
    initialize(grid1, n);

    // Simulation loop
    for (int iter = 0; iter < k; ++iter) {
        // Perform one iteration of the Ising model
        update(grid1, grid2, n);

        // Swap the pointers for the next iteration
        int *temp = grid1;
        grid1 = grid2;
        grid2 = temp;

	// Introduce a delay to slow down the animation
        // Adjust the delay based on your preference
        if (c == 'a') {
        	printState(grid1, n);
		usleep(100000); // 100 milliseconds
    	}
    }

    // Print the final state of the Ising model
    printf("Final state:\n");
    printState(grid1, n);

    // Free allocated memory
    free(grid1);
    free(grid2);

    // Stop the timer for the CPU version
    cpu_end = clock();
    double cpu_time_used = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU Time: %f seconds\n", cpu_time_used);

    return 0;
}

