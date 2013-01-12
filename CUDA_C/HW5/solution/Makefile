NVCC=nvcc
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra

GCC_OPTS=-O3 -Wall -Wextra

histo: compare main.cu reference.cpp solution.o Makefile
	nvcc -o histo main.cu solution.o -O3

solution.o: solution.cu
	nvcc -c solution.cu -arch=sm_20 -O3

compare: compare.cu
	nvcc -o compare compare.cu -O3

clean:
	rm -f *.o compare histo *.bin
