CPU = cpu
CPU_TARGET = cpu.c

GPU = gpu 
GPU_TARGET = gpu.c 

CC = clang

FLAGS_CPU = -mavx2 -mfma -O3 -march=native -fopenmp 
FLAGS_GPU = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1101

$(CPU): $(CPU_TARGET)
	@$(CC) $(FLAGS_CPU) -o $(CPU) $(CPU_TARGET)

$(GPU): $(GPU_TARGET)
	@$(CC) $(FLAGS_GPU) -o $(GPU) $(GPU_TARGET)

clean:
	@ rm cpu gpu