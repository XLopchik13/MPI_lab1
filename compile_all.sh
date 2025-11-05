#!/bin/bash
# Скрипт для компиляции всех программ MPI в MSYS2/MinGW

echo "========================================"
echo "Компиляция программ MPI"
echo "========================================"
echo

echo "--- Задача 1: Вычисление Pi ---"
cd task1_pi
if [ -f Makefile ]; then
    make clean
    make
    if [ $? -eq 0 ]; then
        echo "[OK] task1_pi скомпилирована"
    else
        echo "[ОШИБКА] Не удалось скомпилировать task1_pi"
    fi
else
    echo "[ОШИБКА] Makefile не найден в task1_pi"
fi
cd ..

echo
echo "--- Задача 2: Умножение матрицы на вектор ---"
cd task2_matvec
if [ -f Makefile ]; then
    make clean
    make
    if [ $? -eq 0 ]; then
        echo "[OK] task2_matvec скомпилирована"
    else
        echo "[ОШИБКА] Не удалось скомпилировать task2_matvec"
    fi
else
    echo "[ОШИБКА] Makefile не найден в task2_matvec"
fi
cd ..

echo
echo "========================================"
echo "Компиляция завершена"
echo "========================================"
