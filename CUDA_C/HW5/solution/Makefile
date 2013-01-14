NVCC=nvcc
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra

histo: compare ../student/main.cu ../student/reference.cpp solution.o Makefile
	nvcc -o histo ../student/main.cu solution.o $(NVCC_OPTS)

solution.o: solution.cu
	nvcc -c solution.cu $(NVCC_OPTS)

compare: ../student/compare.cu
	nvcc -o compare ../student/compare.cu $(NVCC_OPTS)

clean:
	rm -f *.o compare histo *.bin
