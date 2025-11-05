#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

double random_double() {
    return (double)rand() / (double)RAND_MAX;
}

int main(int argc, char *argv[]) {
    int rank, size;
    long long total_points, local_points;
    long long local_inside = 0, total_inside = 0;
    double x, y, distance;
    double pi_estimate;
    double start_time, end_time, elapsed_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <number_of_points>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    total_points = atoll(argv[1]);

    local_points = total_points / size;

    if (rank == size - 1) {
        local_points += total_points % size;
    }

    srand(time(NULL) + rank);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (long long i = 0; i < local_points; i++) {
        x = random_double() * 2.0 - 1.0;
        y = random_double() * 2.0 - 1.0;

        distance = x * x + y * y;
        if (distance <= 1.0) {
            local_inside++;
        }
    }

    MPI_Reduce(&local_inside, &total_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    if (rank == 0) {
        pi_estimate = 4.0 * (double)total_inside / (double)total_points;
        
        printf("=== Monte Carlo Pi Estimation ===\n");
        printf("Total points: %lld\n", total_points);
        printf("Points inside circle: %lld\n", total_inside);
        printf("Pi estimate: %.10f\n", pi_estimate);
        printf("Actual Pi: %.10f\n", M_PI);
        printf("Error: %.10f\n", fabs(pi_estimate - M_PI));
        printf("Processes: %d\n", size);
        printf("Execution time: %.6f seconds\n", elapsed_time);

        printf("\nCSV,%d,%lld,%.6f,%.10f\n", size, total_points, elapsed_time, pi_estimate);
    }

    MPI_Finalize();
    return 0;
}
