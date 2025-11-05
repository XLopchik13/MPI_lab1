/**
 * Умножение матрицы на вектор с распределением по БЛОКАМ
 * Матрица разбивается на прямоугольные блоки
 * 
 * Компиляция: mpicc -o matvec_blocks.exe matvec_blocks.c -lm
 * Запуск: mpiexec -n <num_procs> matvec_blocks.exe <matrix_size>
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
    int grid_rows, grid_cols;
    int local_rows, local_cols;
    int block_row, block_col;
    double *matrix = NULL;
    double *vector = NULL;
    double *local_matrix = NULL;
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
    
    // Создание 2D сетки процессов (приближённо квадратной)
    grid_rows = (int)sqrt(size);
    grid_cols = size / grid_rows;
    
    // Если размер не идеально делится, корректируем
    while (grid_rows * grid_cols < size) {
        grid_cols++;
    }
    
    if (grid_rows * grid_cols > size) {
        if (rank == 0) {
            fprintf(stderr, "Warning: Using %d out of %d processes\n", grid_rows * grid_cols, size);
        }
    }
    
    // Определение позиции процесса в сетке
    block_row = rank / grid_cols;
    block_col = rank % grid_cols;
    
    // Если процесс за пределами сетки, он не участвует
    if (rank >= grid_rows * grid_cols) {
        MPI_Finalize();
        return 0;
    }
    
    // Размеры локального блока
    local_rows = N / grid_rows;
    local_cols = N / grid_cols;
    
    int row_remainder = N % grid_rows;
    int col_remainder = N % grid_cols;
    
    if (block_row < row_remainder) local_rows++;
    if (block_col < col_remainder) local_cols++;
    
    // Главный процесс инициализирует данные
    if (rank == 0) {
        matrix = (double*)malloc(N * N * sizeof(double));
        vector = (double*)malloc(N * sizeof(double));
        result = (double*)malloc(N * sizeof(double));
        
        srand(time(NULL));
        initialize_matrix_vector(matrix, vector, N, N);
        
        for (int i = 0; i < N; i++) {
            result[i] = 0.0;
        }
    }
    
    local_matrix = (double*)malloc(local_rows * local_cols * sizeof(double));
    local_vector = (double*)malloc(local_cols * sizeof(double));
    local_result = (double*)malloc(local_rows * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Распределение блоков матрицы и вектора
    if (rank == 0) {
        // Вычисление начальных позиций для каждого процесса
        for (int p = 0; p < grid_rows * grid_cols && p < size; p++) {
            int p_row = p / grid_cols;
            int p_col = p % grid_cols;
            
            int p_local_rows = N / grid_rows;
            int p_local_cols = N / grid_cols;
            if (p_row < row_remainder) p_local_rows++;
            if (p_col < col_remainder) p_local_cols++;
            
            int start_row = p_row * (N / grid_rows) + (p_row < row_remainder ? p_row : row_remainder);
            int start_col = p_col * (N / grid_cols) + (p_col < col_remainder ? p_col : col_remainder);
            
            // Извлечение блока матрицы
            double *block = (double*)malloc(p_local_rows * p_local_cols * sizeof(double));
            for (int i = 0; i < p_local_rows; i++) {
                for (int j = 0; j < p_local_cols; j++) {
                    block[i * p_local_cols + j] = matrix[(start_row + i) * N + (start_col + j)];
                }
            }
            
            // Извлечение части вектора
            double *vec_part = (double*)malloc(p_local_cols * sizeof(double));
            for (int j = 0; j < p_local_cols; j++) {
                vec_part[j] = vector[start_col + j];
            }
            
            if (p == 0) {
                for (int i = 0; i < p_local_rows * p_local_cols; i++) {
                    local_matrix[i] = block[i];
                }
                for (int i = 0; i < p_local_cols; i++) {
                    local_vector[i] = vec_part[i];
                }
            } else {
                MPI_Send(block, p_local_rows * p_local_cols, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Send(vec_part, p_local_cols, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
            }
            
            free(block);
            free(vec_part);
        }
    } else {
        MPI_Recv(local_matrix, local_rows * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_vector, local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Локальное умножение блока на часть вектора
    for (int i = 0; i < local_rows; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            local_result[i] += local_matrix[i * local_cols + j] * local_vector[j];
        }
    }
    
    // Сбор результатов по строкам (процессы в одной строке сетки суммируют свои результаты)
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, block_row, rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, block_col, rank, &col_comm);
    
    double *row_result = NULL;
    if (block_col == 0) {
        row_result = (double*)malloc(local_rows * sizeof(double));
    }
    
    MPI_Reduce(local_result, row_result, local_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    
    // Главный столбец собирает результаты (используем col_comm для процессов с block_col==0)
    if (block_col == 0) {
        if (rank == 0) {
            int *recvcounts = (int*)malloc(grid_rows * sizeof(int));
            int *displs = (int*)malloc(grid_rows * sizeof(int));
            
            int offset = 0;
            for (int r = 0; r < grid_rows; r++) {
                int rows_for_block = N / grid_rows;
                if (r < row_remainder) rows_for_block++;
                recvcounts[r] = rows_for_block;
                displs[r] = offset;
                offset += rows_for_block;
            }
            
            MPI_Gatherv(row_result, local_rows, MPI_DOUBLE,
                       result, recvcounts, displs, MPI_DOUBLE,
                       0, col_comm);
            
            free(recvcounts);
            free(displs);
        } else {
            MPI_Gatherv(row_result, local_rows, MPI_DOUBLE,
                       NULL, NULL, NULL, MPI_DOUBLE,
                       0, col_comm);
        }
        free(row_result);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    if (rank == 0) {
        printf("=== Matrix-Vector Multiplication (Block Distribution) ===\n");
        printf("Matrix size: %d x %d\n", N, N);
        printf("Grid: %d x %d\n", grid_rows, grid_cols);
        printf("Processes: %d\n", size);
        printf("Execution time: %.6f seconds\n", elapsed_time);
        printf("\nCSV,blocks,%d,%d,%.6f\n", size, N, elapsed_time);
        
        free(matrix);
        free(vector);
        free(result);
    }
    
    free(local_matrix);
    free(local_vector);
    free(local_result);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
