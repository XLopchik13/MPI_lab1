# Makefile для компиляции программ умножения матрицы на вектор (MS-MPI + MinGW)
# Используются переменные MSMPI_INC / MSMPI_LIB при наличии.

MSMPI_INC ?= C:/Program Files (x86)/Microsoft SDKs/MPI/Include
MSMPI_LIB ?= C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64

CC = gcc
CFLAGS = -O2 -Wall -I"$(MSMPI_INC)"
LDFLAGS = -L"$(MSMPI_LIB)"
LIBS = -lmsmpi -lm

TARGETS = matvec_rows.exe matvec_cols.exe matvec_blocks.exe

all: $(TARGETS)

matvec_rows.exe: matvec_rows.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

matvec_cols.exe: matvec_cols.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

matvec_blocks.exe: matvec_blocks.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS) *.o

test:
	mpiexec -n 4 ./matvec_rows.exe 1000
	mpiexec -n 4 ./matvec_cols.exe 1000
	mpiexec -n 4 ./matvec_blocks.exe 1000

.PHONY: all clean test
