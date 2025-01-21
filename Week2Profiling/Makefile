PROGRAM = main
SRC = main.c

LEARN = learn
L_SRC = learnvectorization.c

FLAMEGRAPH_DIR = ../Flamegraph
PERF_OUTPUT = perf.data
PROFILING_SCRIPT = $(FLAMEGRAPH_DIR)/stackcollapse-perf.pl
FLAMEGRAPH_SCRIPT = $(FLAMEGRAPH_DIR)/flamegraph.pl

# Debugging and profiling enabled
CFLAGS = -g -pg -mavx2 -mfma

all: $(PROGRAM) $(LEARN)

$(PROGRAM): $(SRC)
	@$(CC) $(CFLAGS) -o $(PROGRAM) $(SRC)

$(LEARN): $(L_SRC)
	@$(CC) $(CFLAGS) -o $(LEARN) $(L_SRC)

run: $(PROGRAM)
	@echo -e "\033[1;32mRunning the program with perf profiling...\033[0m"
	@perf record -F 99 -a -g -- ./$(PROGRAM)

# Generate flame graph after running the program
profile: run
	@echo -e "\033[1;34mConverting perf data to stack traces...\033[0m"

	@perf script > out.perf
	
	@echo -e "\033[1;34mCreating flame graph...\033[0m"

	@$(PROFILING_SCRIPT) out.perf | $(FLAMEGRAPH_SCRIPT) > flamegraph.svg
	
	@echo -e "\033[1;32mFlame graph generated as flamegraph.svg.\033[0m"
	@echo -e "\033[1;33mGenerating gprof report...\033[0m"

	@gprof $(PROGRAM) gmon.out > report.txt
	
	@echo -e "\033[1;32mGprof report saved as report.txt.\033[0m"

# Clean up generated files
clean:
	@echo -e "\033[1;31mCleaning up generated files...\033[0m"

	@rm -f $(PROGRAM) $(PERF_OUTPUT) out.perf flamegraph.svg gmon.out report.txt learn *.o callgrind*

	@echo -e "\033[1;31mCleaned up generated files.\033[0m"




