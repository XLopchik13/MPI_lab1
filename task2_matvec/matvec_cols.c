/**
 * Умножение матрицы на вектор с распределением по СТОЛБЦАМ
 * Каждый процесс обрабатывает свою часть столбцов матрицы
 * 
 * Компиляция: mpicc -o matvec_cols.exe matvec_cols.c -lm
 * Запуск: mpiexec -n <num_procs> matvec_cols.exe <matrix_size>
 */

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
    int local_cols;
    double *matrix = NULL;
    double *vector = NULL;
    double *local_vector = NULL;
    double *local_result = NULL;
    double *result = NULL;
    double start_time, end_time, elapsed_time;
    
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
    
    // Распределение столбцов
    local_cols = N / size;
    int remainder = N % size;
    
    if (rank < remainder) {
        local_cols++;
    }
    
    // Главный процесс инициализирует данные
    if (rank == 0) {
        matrix = (double*)malloc(N * N * sizeof(double));
        vector = (double*)malloc(N * sizeof(double));
        result = (double*)malloc(N * sizeof(double));
        
        srand(time(NULL));
        initialize_matrix_vector(matrix, vector, N, N);
        
        // Инициализация результата нулями
        for (int i = 0; i < N; i++) {
            result[i] = 0.0;
        }
    }
    
    // Локальные массивы
    local_vector = (double*)malloc(local_cols * sizeof(double));
    local_result = (double*)malloc(N * sizeof(double));
    
    // Временная матрица для столбцов
    double *local_matrix_cols = (double*)malloc(N * local_cols * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Распределение части вектора
    int *sendcounts_vec = NULL;
    int *displs_vec = NULL;
    
    if (rank == 0) {
        sendcounts_vec = (int*)malloc(size * sizeof(int));
        displs_vec = (int*)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int cols_for_proc = N / size;
            if (i < remainder) cols_for_proc++;
            sendcounts_vec[i] = cols_for_proc;
            displs_vec[i] = offset;
            offset += cols_for_proc;
        }
    }
    
    MPI_Scatterv(vector, sendcounts_vec, displs_vec, MPI_DOUBLE,
                 local_vector, local_cols, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Распределение столбцов матрицы
    // Каждый процесс получает свои столбцы
    if (rank == 0) {
        // Отправка столбцов каждому процессу
        for (int p = 0; p < size; p++) {
            int start_col = displs_vec[p];
            int num_cols = sendcounts_vec[p];
            
            double *temp = (double*)malloc(N * num_cols * sizeof(double));
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < num_cols; j++) {
                    temp[i * num_cols + j] = matrix[i * N + (start_col + j)];
                }
            }
            
            if (p == 0) {
                for (int i = 0; i < N * num_cols; i++) {
                    local_matrix_cols[i] = temp[i];
                }
            } else {
                MPI_Send(temp, N * num_cols, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
            
            free(temp);
        }
    } else {
        MPI_Recv(local_matrix_cols, N * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Локальное вычисление частичного результата
    for (int i = 0; i < N; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            local_result[i] += local_matrix_cols[i * local_cols + j] * local_vector[j];
        }
    }
    
    // Суммирование частичных результатов
    MPI_Reduce(local_result, result, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    if (rank == 0) {
        printf("=== Matrix-Vector Multiplication (Column Distribution) ===\n");
        printf("Matrix size: %d x %d\n", N, N);
        printf("Processes: %d\n", size);
        printf("Execution time: %.6f seconds\n", elapsed_time);
        printf("\nCSV,cols,%d,%d,%.6f\n", size, N, elapsed_time);
        
        free(matrix);
        free(vector);
        free(result);
        free(sendcounts_vec);
        free(displs_vec);
    }
    
    free(local_vector);
    free(local_result);
    free(local_matrix_cols);
    
    MPI_Finalize();
    return 0;
}
