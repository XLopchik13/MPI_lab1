@echo off
REM Скрипт для компиляции всех программ MPI (Windows PowerShell/Command Prompt)
REM Важно: убедитесь, что установлен MinGW (MSYS2) и доступны gcc и make в PATH.
REM Добавьте при необходимости: set PATH=C:\msys64\usr\bin;C:\msys64\mingw64\bin;%PATH%
chcp 65001 >nul

echo ========================================
echo Компиляция программ MPI
echo ========================================
echo.

echo --- Задача 1: Вычисление Pi ---
pushd task1_pi >nul 2>&1
if not exist Makefile (
    echo [ОШИБКА] Makefile не найден в task1_pi
) else (
    make clean >nul 2>&1
    make
    if exist pi_monte_carlo.exe (
        echo [OK] task1_pi собрана
    ) else (
        echo [ОШИБКА] Ошибка сборки task1_pi (проверьте пути MSMPI_INC / MSMPI_LIB)
    )
)
popd >nul 2>&1

echo.
echo --- Задача 2: Умножение матрицы на вектор ---
pushd task2_matvec >nul 2>&1
if not exist Makefile (
    echo [ОШИБКА] Makefile не найден в task2_matvec
) else (
    make clean >nul 2>&1
    make
    for %%F in (matvec_rows.exe matvec_cols.exe matvec_blocks.exe) do (
        if exist %%F (
            echo [OK] %%F собрана
        ) else (
            echo [ОШИБКА] Ошибка сборки %%F
        )
    )
)
popd >nul 2>&1

echo.
echo ========================================
echo Компиляция завершена
echo.
echo Если сборка не удалась, проверьте что заданы переменные окружения:
echo   MSMPI_INC  (обычно: C:\Program Files (x86)\Microsoft SDKs\MPI\Include)
echo   MSMPI_LIB  (обычно: C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64)
echo Можно переопределить их прямо при вызове: make MSMPI_INC="C:/.../Include" MSMPI_LIB="C:/.../Lib/x64"
echo ========================================
pause
