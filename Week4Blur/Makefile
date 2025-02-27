SRC = blur.c

TARGET_UNOP = blur_unop
TARGET_OP = blur_op
TARGET_SIMD = blur_simd
TARGET_ALL = blur_


CFLAGS_UNOP = -mavx2 -mfma -O2 -march=native -fopenmp -funroll-loops -lm -g -DUNOP
CFLAGS_OP = -mavx2 -mfma -O2 -march=native -fopenmp -funroll-loops -lm -g -DOP
CFLAGS_SIMD = -mavx2 -mfma -O2 -march=native -fopenmp -funroll-loops -lm -g -DOP_SIMD
CFLAGS_TEST_ALL = -mavx2 -mfma -O2 -march=native -fopenmp -funroll-loops -lm -g -DUNOP -DOP -DOP_SIMD

$(TARGET_UNOP): $(SRC)
	@$(CC) $(CFLAGS_UNOP) $(SRC) -o $(TARGET_UNOP)

$(TARGET_OP): $(SRC)
	@$(CC) $(CFLAGS_OP) $(SRC) -o $(TARGET_OP)

$(TARGET_SIMD): $(SRC)
	@$(CC) $(CFLAGS_SIMD) $(SRC) -o $(TARGET_SIMD)

$(TARGET_ALL): $(SRC)
	@$(CC) $(CFLAGS_TEST_ALL) $(SRC) -o $(TARGET_ALL)

clean: 
	@rm blur_* images/output.*