import subprocess
import os
import sys
import time
from pathlib import Path
import shutil

CONFIG = {
    'task1_pi': {
        'executable': '../task1_pi/pi_monte_carlo.exe',
        'processes': [1, 2, 4, 8],
        'parameters': [
            1000000,
            5000000,
            10000000,
            20000000,
        ],
        'repeats': 3,
    },
    'task2_matvec': {
        'executables': {
            'rows': '../task2_matvec/matvec_rows.exe',
            'cols': '../task2_matvec/matvec_cols.exe',
            'blocks': '../task2_matvec/matvec_blocks.exe',
        },
        'processes': [1, 2, 4, 8],
        'parameters': [
            1000,
            2000,
            5000,
            10000,
            15000,
        ],
        'repeats': 3,
    }
}

def run_command(command, timeout=300):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return None, "Timeout", -1
    except Exception as e:
        return None, str(e), -1

def parse_csv_output(output):
    lines = output.strip().split('\n')
    for line in lines:
        if line.startswith('CSV,'):
            return line
    return None

def run_task1_experiments(output_dir, mpiexec_available):
    print("=" * 60)
    print("ЗАДАЧА 1: Вычисление π методом Монте-Карло")
    print("=" * 60)
    
    results_file = output_dir / 'task1_results.csv'
    
    with open(results_file, 'w') as f:
        f.write('processes,points,time,pi_estimate,repeat\n')
    
    config = CONFIG['task1_pi']
    total_runs = len(config['processes']) * len(config['parameters']) * config['repeats']
    current_run = 0
    
    for points in config['parameters']:
        for procs in config['processes']:
            for repeat in range(config['repeats']):
                current_run += 1
                print(f"[{current_run}/{total_runs}] Процессы: {procs}, Точки: {points}, Повтор: {repeat+1}/{config['repeats']}")
                
                if not mpiexec_available:
                    if procs != 1:
                        print(f"  ⏭ Пропуск (mpiexec недоступен, процессы={procs})")
                        continue
                    # Однопроцессный запуск без mpiexec
                    cmd = f"{config['executable']} {points}"
                else:
                    cmd = f"mpiexec -n {procs} {config['executable']} {points}"
                stdout, stderr, returncode = run_command(cmd)
                
                if returncode != 0:
                    print(f"ОШИБКА: {stderr}")
                    continue
                
                csv_line = parse_csv_output(stdout)
                if csv_line:
                    # CSV,size,points,time,pi_estimate
                    parts = csv_line.split(',')
                    if len(parts) >= 5:
                        procs_val = parts[1]
                        points_val = parts[2]
                        time_val = parts[3]
                        pi_val = parts[4]
                        
                        with open(results_file, 'a') as f:
                            f.write(f"{procs_val},{points_val},{time_val},{pi_val},{repeat+1}\n")
                        
                        print(f"  ✓ Время: {time_val}s, π ≈ {pi_val}")
                else:
                    print(f"  ⚠ Не удалось извлечь результаты")
                
                time.sleep(0.5)  # Пауза между запусками
    
    print(f"\n✓ Результаты сохранены в {results_file}\n")

def run_task2_experiments(output_dir, mpiexec_available):
    """Запуск экспериментов для задачи 2 (матрица × вектор)"""
    print("=" * 60)
    print("ЗАДАЧА 2: Умножение матрицы на вектор")
    print("=" * 60)
    
    results_file = output_dir / 'task2_results.csv'
    
    with open(results_file, 'w') as f:
        f.write('algorithm,processes,matrix_size,time,repeat\n')
    
    config = CONFIG['task2_matvec']
    total_runs = len(config['executables']) * len(config['processes']) * len(config['parameters']) * config['repeats']
    current_run = 0
    
    for algo_name, executable in config['executables'].items():
        print(f"\n--- Алгоритм: {algo_name.upper()} ---")
        
        for size in config['parameters']:
            for procs in config['processes']:
                for repeat in range(config['repeats']):
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] {algo_name}, Процессы: {procs}, Размер: {size}x{size}, Повтор: {repeat+1}/{config['repeats']}")
                    
                    if not mpiexec_available:
                        if procs != 1:
                            print(f"  ⏭ Пропуск (mpiexec недоступен, процессы={procs})")
                            continue
                        cmd = f"{executable} {size}"
                    else:
                        cmd = f"mpiexec -n {procs} {executable} {size}"
                    stdout, stderr, returncode = run_command(cmd, timeout=900)
                    
                    if returncode != 0:
                        print(f"ОШИБКА: {stderr}")
                        continue
                    
                    csv_line = parse_csv_output(stdout)
                    if csv_line:
                        # CSV,algorithm,size,matrix_size,time
                        parts = csv_line.split(',')
                        if len(parts) >= 5:
                            algo = parts[1]
                            procs_val = parts[2]
                            size_val = parts[3]
                            time_val = parts[4]
                            
                            with open(results_file, 'a') as f:
                                f.write(f"{algo},{procs_val},{size_val},{time_val},{repeat+1}\n")
                            
                            print(f"  ✓ Время: {time_val}s")
                    else:
                        print(f"  ⚠ Не удалось извлечь результаты")
                    
                    time.sleep(0.5)
    
    print(f"\n✓ Результаты сохранены в {results_file}\n")

def main():
    # Создание директории для результатов
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("АВТОМАТИЧЕСКИЙ ЗАПУСК ЭКСПЕРИМЕНТОВ MPI")
    print("=" * 60 + "\n")
    
    # Проверка наличия исполняемых файлов
    print("Проверка исполняемых файлов...")
    
    missing_files = []
    
    if not Path(CONFIG['task1_pi']['executable']).exists():
        missing_files.append(CONFIG['task1_pi']['executable'])
    
    for exe in CONFIG['task2_matvec']['executables'].values():
        if not Path(exe).exists():
            missing_files.append(exe)
    
    if missing_files:
        print("\nОШИБКА: Не найдены следующие файлы:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nСкомпилируйте программы перед запуском экспериментов:")
        print("  cd task1_pi && make")
        print("  cd task2_matvec && make")
        sys.exit(1)
    
    print("✓ Все исполняемые файлы найдены\n")
    
    start_time = time.time()
    
    # Проверка наличия mpiexec
    mpiexec_available = shutil.which("mpiexec") is not None
    if not mpiexec_available:
        print("⚠ mpiexec не найден в PATH. Будут выполнены только однопроцессные запуски (p=1).")
        print("  Установите MS-MPI Runtime и добавьте C:\\Program Files\\Microsoft MPI\\Bin в PATH.")
        print("  Пример временно в PowerShell: $env:Path = 'C:/Program Files/Microsoft MPI/Bin;' + $env:Path")
        print()

    # Запуск экспериментов
    try:
        run_task1_experiments(output_dir, mpiexec_available)
        run_task2_experiments(output_dir, mpiexec_available)
    except KeyboardInterrupt:
        print("\n\n⚠ Эксперименты прерваны пользователем")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    print("=" * 60)
    print(f"✓ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"Общее время: {elapsed:.1f} секунд ({elapsed/60:.1f} минут)")
    print("=" * 60)
    print(f"\nРезультаты сохранены в директории: {output_dir.absolute()}")
    print("\nСледующий шаг: запустите analyze_results.py для построения графиков")

if __name__ == '__main__':
    main()
