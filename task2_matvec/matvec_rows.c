#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

void initialize_matrix_vector(double *matrix, double *vector, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)(rand() % 100) / 10.0;
    }
    for (int i = 0; i < cols; i++) {
        vector[i] = (double)(rand() % 100) / 10.0;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    int local_rows;
    double *matrix = NULL;
    double *vector = NULL;
    double *local_matrix = NULL;
    double *local_result = NULL;
    double *result = NULL;
    double start_time, end_time, elapsed_time;
    int *sendcounts = NULL;
    int *displs = NULL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    N = atoi(argv[1]);

    local_rows = N / size;
    int remainder = N % size;

    if (rank < remainder) {
        local_rows++;
    }

    if (rank == 0) {
        matrix = (double*)malloc(N * N * sizeof(double));
        vector = (double*)malloc(N * sizeof(double));
        result = (double*)malloc(N * sizeof(double));
        
        srand(time(NULL));
        initialize_matrix_vector(matrix, vector, N, N);

        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_proc = N / size;
            if (i < remainder) rows_for_proc++;
            sendcounts[i] = rows_for_proc * N;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    local_matrix = (double*)malloc(local_rows * N * sizeof(double));
    local_result = (double*)malloc(local_rows * sizeof(double));
    vector = (double*)malloc(N * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    MPI_Bcast(vector, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE,
                 local_matrix, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < N; j++) {
            local_result[i] += local_matrix[i * N + j] * vector[j];
        }
    }

    if (rank == 0) {
        int *recvcounts = (int*)malloc(size * sizeof(int));
        int *recvdispls = (int*)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_proc = N / size;
            if (i < remainder) rows_for_proc++;
            recvcounts[i] = rows_for_proc;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
        
        MPI_Gatherv(local_result, local_rows, MPI_DOUBLE,
                    result, recvcounts, recvdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        free(recvcounts);
        free(recvdispls);
    } else {
        MPI_Gatherv(local_result, local_rows, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    if (rank == 0) {
        printf("=== Matrix-Vector Multiplication (Row Distribution) ===\n");
        printf("Matrix size: %d x %d\n", N, N);
        printf("Processes: %d\n", size);
        printf("Execution time: %.6f seconds\n", elapsed_time);
        printf("\nCSV,rows,%d,%d,%.6f\n", size, N, elapsed_time);
        
        free(matrix);
        free(result);
        free(sendcounts);
        free(displs);
    }
    
    free(vector);
    free(local_matrix);
    free(local_result);
    
    MPI_Finalize();
    return 0;
}
