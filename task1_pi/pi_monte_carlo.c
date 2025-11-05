/**
 * Параллельное вычисление числа π методом Монте-Карло
 * Использует MPI для распределённых вычислений
 * 
 * Компиляция: mpicc -o pi_monte_carlo.exe pi_monte_carlo.c -lm
 * Запуск: mpiexec -n <num_procs> pi_monte_carlo.exe <num_points>
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

// Генерация случайного числа в диапазоне [0, 1]
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
    
    // Проверка аргументов
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <number_of_points>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    total_points = atoll(argv[1]);
    
    // Распределение точек между процессами
    local_points = total_points / size;
    
    // Последний процесс обрабатывает оставшиеся точки
    if (rank == size - 1) {
        local_points += total_points % size;
    }
    
    // Уникальный seed для каждого процесса
    srand(time(NULL) + rank);
    
    // Синхронизация перед началом измерения времени
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Метод Монте-Карло: подсчёт точек внутри окружности
    for (long long i = 0; i < local_points; i++) {
        // Случайная точка в квадрате [-1, 1] × [-1, 1]
        x = random_double() * 2.0 - 1.0;
        y = random_double() * 2.0 - 1.0;
        
        // Проверка попадания в единичную окружность
        distance = x * x + y * y;
        if (distance <= 1.0) {
            local_inside++;
        }
    }
    
    // Сбор результатов от всех процессов
    MPI_Reduce(&local_inside, &total_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Синхронизация после вычислений
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    // Вывод результатов (только главный процесс)
    if (rank == 0) {
        // π ≈ 4 * (точки в круге / общее число точек)
        pi_estimate = 4.0 * (double)total_inside / (double)total_points;
        
        printf("=== Monte Carlo Pi Estimation ===\n");
        printf("Total points: %lld\n", total_points);
        printf("Points inside circle: %lld\n", total_inside);
        printf("Pi estimate: %.10f\n", pi_estimate);
        printf("Actual Pi: %.10f\n", M_PI);
        printf("Error: %.10f\n", fabs(pi_estimate - M_PI));
        printf("Processes: %d\n", size);
        printf("Execution time: %.6f seconds\n", elapsed_time);
        
        // CSV формат для автоматической обработки
        printf("\nCSV,%d,%lld,%.6f,%.10f\n", size, total_points, elapsed_time, pi_estimate);
    }
    
    MPI_Finalize();
    return 0;
}
