import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_task1_results(results_dir):
    csv_file = results_dir / 'task1_results.csv'

    if not csv_file.exists():
        print(f"Файл результатов не найден: {csv_file}")
        return None

    df = pd.read_csv(csv_file)

    df_avg = df.groupby(['processes', 'points']).agg({
        'time': ['mean', 'std'],
        'pi_estimate': 'mean'
    }).reset_index()
    
    df_avg.columns = ['processes', 'points', 'time_mean', 'time_std', 'pi_estimate']
    
    return df_avg

def load_task2_results(results_dir):
    csv_file = results_dir / 'task2_results.csv'

    if not csv_file.exists():
        print(f"Файл результатов не найден: {csv_file}")
        return None

    df = pd.read_csv(csv_file)

    df_avg = df.groupby(['algorithm', 'processes', 'matrix_size']).agg({
        'time': ['mean', 'std']
    }).reset_index()

    df_avg.columns = ['algorithm', 'processes', 'matrix_size', 'time_mean', 'time_std']

    return df_avg

def calculate_speedup_efficiency(df, group_cols, time_col='time_mean'):
    results = []
    
    for group_vals, group_df in df.groupby(group_cols):
        group_df = group_df.sort_values('processes')

        t1_row = group_df[group_df['processes'] == 1]
        
        if len(t1_row) == 0:
            continue

        t1 = t1_row[time_col].values[0]

        for _, row in group_df.iterrows():
            p = row['processes']
            tp = row[time_col]

            speedup = t1 / tp if tp > 0 else 0
            efficiency = speedup / p if p > 0 else 0

            result_row = row.to_dict()
            result_row['speedup'] = speedup
            result_row['efficiency'] = efficiency
            result_row['t1'] = t1

            results.append(result_row)

    return pd.DataFrame(results)

def plot_task1_results(df, output_dir):
    print("\n--- Построение графиков для Задачи 1 ---")

    df_metrics = calculate_speedup_efficiency(df, ['points'])
    
    unique_points = sorted(df_metrics['points'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for points in unique_points:
        data = df_metrics[df_metrics['points'] == points].sort_values('processes')
        ax.plot(data['processes'], data['time_mean'], 
                marker='o', label=f'{points:,} points')
        ax.fill_between(data['processes'], 
                        data['time_mean'] - data['time_std'],
                        data['time_mean'] + data['time_std'],
                        alpha=0.2)
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Task 1: Pi Estimation - Execution Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'task1_time.png', dpi=300)
    print(f"Сохранён: task1_time.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    for points in unique_points:
        data = df_metrics[df_metrics['points'] == points].sort_values('processes')
        ax.plot(data['processes'], data['speedup'], 
                marker='s', label=f'{points:,} points')

    max_procs = df_metrics['processes'].max()
    ax.plot([1, max_procs], [1, max_procs], 'k--', label='Ideal Speedup', alpha=0.5)
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Task 1: Pi Estimation - Speedup', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'task1_speedup.png', dpi=300)
    print(f"Сохранён: task1_speedup.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for points in unique_points:
        data = df_metrics[df_metrics['points'] == points].sort_values('processes')
        ax.plot(data['processes'], data['efficiency'] * 100, 
                marker='^', label=f'{points:,} points')

    ax.axhline(y=100, color='k', linestyle='--', label='100% Efficiency', alpha=0.5)
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Task 1: Pi Estimation - Efficiency', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(output_dir / 'task1_efficiency.png', dpi=300)
    print(f"Сохранён: task1_efficiency.png")
    plt.close()

    df_metrics.to_csv(output_dir / 'task1_metrics.csv', index=False)
    print(f"Сохранена таблица метрик: task1_metrics.csv")

def plot_task2_results(df, output_dir):
    print("\n--- Построение графиков для Задачи 2 ---")
    
    algorithms = sorted(df['algorithm'].unique())
    
    for size in sorted(df['matrix_size'].unique()):
        df_size = df[df['matrix_size'] == size]
        df_metrics = calculate_speedup_efficiency(df_size, ['algorithm', 'matrix_size'])

        fig, ax = plt.subplots(figsize=(10, 6))

        for algo in algorithms:
            data = df_metrics[df_metrics['algorithm'] == algo].sort_values('processes')
            ax.plot(data['processes'], data['time_mean'], 
                    marker='o', label=algo.capitalize())
            ax.fill_between(data['processes'], 
                            data['time_mean'] - data['time_std'],
                            data['time_mean'] + data['time_std'],
                            alpha=0.2)

        ax.set_xlabel('Number of Processes', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title(f'Task 2: Matrix-Vector ({size}x{size}) - Execution Time', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'task2_time_{size}.png', dpi=300)
        print(f"Сохранён: task2_time_{size}.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))

        for algo in algorithms:
            data = df_metrics[df_metrics['algorithm'] == algo].sort_values('processes')
            ax.plot(data['processes'], data['speedup'], 
                    marker='s', label=algo.capitalize())

        max_procs = df_metrics['processes'].max()
        ax.plot([1, max_procs], [1, max_procs], 'k--', label='Ideal Speedup', alpha=0.5)

        ax.set_xlabel('Number of Processes', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title(f'Task 2: Matrix-Vector ({size}x{size}) - Speedup', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'task2_speedup_{size}.png', dpi=300)
        print(f"✓ Сохранён: task2_speedup_{size}.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))

        for algo in algorithms:
            data = df_metrics[df_metrics['algorithm'] == algo].sort_values('processes')
            ax.plot(data['processes'], data['efficiency'] * 100, 
                    marker='^', label=algo.capitalize())

        ax.axhline(y=100, color='k', linestyle='--', label='100% Efficiency', alpha=0.5)

        ax.set_xlabel('Number of Processes', fontsize=12)
        ax.set_ylabel('Efficiency (%)', fontsize=12)
        ax.set_title(f'Task 2: Matrix-Vector ({size}x{size}) - Efficiency', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)
        plt.tight_layout()
        plt.savefig(output_dir / f'task2_efficiency_{size}.png', dpi=300)
        print(f"Сохранён: task2_efficiency_{size}.png")
        plt.close()

        df_metrics.to_csv(output_dir / f'task2_metrics_{size}.csv', index=False)
        print(f"Сохранена таблица метрик: task2_metrics_{size}.csv")

    for algo in algorithms:
        df_algo = df[df['algorithm'] == algo]
        df_metrics_algo = calculate_speedup_efficiency(df_algo, ['algorithm', 'matrix_size'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for size in sorted(df_algo['matrix_size'].unique()):
            data = df_metrics_algo[df_metrics_algo['matrix_size'] == size].sort_values('processes')
            ax1.plot(data['processes'], data['speedup'], 
                    marker='o', label=f'{size}x{size}')
            ax2.plot(data['processes'], data['efficiency'] * 100, 
                    marker='s', label=f'{size}x{size}')

        max_procs = df_metrics_algo['processes'].max()
        ax1.plot([1, max_procs], [1, max_procs], 'k--', label='Ideal', alpha=0.5)
        ax2.axhline(y=100, color='k', linestyle='--', label='100%', alpha=0.5)

        ax1.set_xlabel('Number of Processes', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.set_title(f'Speedup - {algo.capitalize()}', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Number of Processes', fontsize=12)
        ax2.set_ylabel('Efficiency (%)', fontsize=12)
        ax2.set_title(f'Efficiency - {algo.capitalize()}', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 110)

        plt.tight_layout()
        plt.savefig(output_dir / f'task2_comparison_{algo}.png', dpi=300)
        print(f"Сохранён: task2_comparison_{algo}.png")
        plt.close()

def generate_summary_report(results_dir, output_dir):
    report_file = output_dir / 'summary_report.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ОТЧЁТ ПО РЕЗУЛЬТАТАМ ЭКСПЕРИМЕНТОВ MPI\n")
        f.write("=" * 70 + "\n\n")

        f.write("ЗАДАЧА 1: Вычисление π методом Монте-Карло\n")
        f.write("-" * 70 + "\n\n")

        task1_metrics = pd.read_csv(output_dir / 'task1_metrics.csv')

        for points in sorted(task1_metrics['points'].unique()):
            data = task1_metrics[task1_metrics['points'] == points].sort_values('processes')
            f.write(f"Количество точек: {points:,}\n")
            f.write(f"{'Процессы':<12} {'Время (s)':<12} {'Ускорение':<12} {'Эффект. (%)':<12}\n")

            for _, row in data.iterrows():
                f.write(f"{int(row['processes']):<12} {row['time_mean']:<12.4f} "
                       f"{row['speedup']:<12.2f} {row['efficiency']*100:<12.1f}\n")
            f.write("\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ЗАДАЧА 2: Умножение матрицы на вектор\n")
        f.write("-" * 70 + "\n\n")

        for size in [1000, 2000, 3000, 5000]:
            metrics_file = output_dir / f'task2_metrics_{size}.csv'
            if not metrics_file.exists():
                continue

            task2_metrics = pd.read_csv(metrics_file)

            f.write(f"\nРазмер матрицы: {size}x{size}\n")
            f.write("=" * 70 + "\n")

            for algo in sorted(task2_metrics['algorithm'].unique()):
                data = task2_metrics[task2_metrics['algorithm'] == algo].sort_values('processes')
                f.write(f"\nАлгоритм: {algo.upper()}\n")
                f.write(f"{'Процессы':<12} {'Время (s)':<12} {'Ускорение':<12} {'Эффект. (%)':<12}\n")

                for _, row in data.iterrows():
                    f.write(f"{int(row['processes']):<12} {row['time_mean']:<12.4f} "
                           f"{row['speedup']:<12.2f} {row['efficiency']*100:<12.1f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ВЫВОДЫ\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. Задача 1 (π):\n")
        f.write("   - Метод Монте-Карло хорошо параллелится\n")
        f.write("   - Ускорение близко к линейному при небольшом числе процессов\n")
        f.write("   - Эффективность снижается с ростом числа процессов из-за накладных расходов\n\n")
        
        f.write("2. Задача 2 (матрица × вектор):\n")
        f.write("   - Разбиение по строкам: наилучшая производительность для больших матриц\n")
        f.write("   - Разбиение по столбцам: дополнительные накладные расходы на коммуникацию\n")
        f.write("   - Разбиение по блокам: баланс между нагрузкой и коммуникацией\n")
        f.write("   - Эффективность зависит от размера матрицы и числа процессов\n\n")

    print(f"\nСохранён отчёт: summary_report.txt")

def main():
    results_dir = Path('../results')

    if not results_dir.exists():
        print(f"Директория результатов не найдена: {results_dir}")
        print("Сначала запустите run_experiments.py")
        sys.exit(1)

    print("=" * 70)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ И ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 70)

    df_task1 = load_task1_results(results_dir)
    if df_task1 is not None:
        plot_task1_results(df_task1, results_dir)

    df_task2 = load_task2_results(results_dir)
    if df_task2 is not None:
        plot_task2_results(df_task2, results_dir)

    if df_task1 is not None or df_task2 is not None:
        generate_summary_report(results_dir, results_dir)
    
    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 70)
    print(f"\nВсе графики и отчёты сохранены в: {results_dir.absolute()}")

if __name__ == '__main__':
    main()
