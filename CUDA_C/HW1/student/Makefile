NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include


OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra

GCC_OPTS=-O3 -Wall -Wextra

student: compare HW1.o main.o student_func.o
	$(NVCC) -o hw main.o HW1.o student_func.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)

HW1.o: HW1.cpp utils.h Makefile
	g++ -c HW1.cpp $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -I $(CUDA_INCLUDEPATH)

main.o: main.cu timer.h utils.h Makefile
	nvcc -c main.cu $(NVCC_OPTS)

student_func.o: student_func.cu reference_calc.cpp utils.h
	nvcc -c student_func.cu $(NVCC_OPTS)

compare: compare.cpp utils.h Makefile
	g++ -o compare compare.cpp -I $(OPENCV_INCLUDEPATH) -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o *.png hw1 compare
