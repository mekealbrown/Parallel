SRC = blur.c
TARGET = blur

CFLAGS = -mavx2 -mfma -O2 -march=native -fopenmp -funroll-loops -lm

$(TARGET): $(SRC)
	@$(CC) $(CFLAGS) $(SRC) -o $(TARGET)


clean: 
	@rm blur images/output.*