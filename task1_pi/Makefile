MSMPI_INC ?= C:/Program Files (x86)/Microsoft SDKs/MPI/Include
MSMPI_LIB ?= C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64

CC = gcc
CFLAGS = -O2 -Wall -I"$(MSMPI_INC)"
LDFLAGS = -L"$(MSMPI_LIB)"
LIBS = -lmsmpi -lm
TARGET = pi_monte_carlo.exe
SRC = pi_monte_carlo.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET) *.o

test:
	mpiexec -n 4 ./$(TARGET) 1000000

.PHONY: all clean test
