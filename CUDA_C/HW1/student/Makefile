NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
CUDA_LIBPATH=/usr/local/cuda-5.0/lib64

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wno-unused -Xcompiler -Wno-uninitialized -w

GCC_OPTS=-O3 -Wno-unused -Wno-uninitialized

student: compare main.o student_func.o Makefile
	$(NVCC) -o hw main.o student_func.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)

main.o: main.cpp timer.h utils.h HW1.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

student_func.o: student_func.cu reference_calc.cpp utils.h
	nvcc -c student_func.cu $(NVCC_OPTS)

compare: compare.cpp utils.h Makefile
	g++ -o compare compare.cpp -lcudart -I $(OPENCV_INCLUDEPATH) -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -L $(CUDA_LIBPATH)

clean:
	rm -f *.o *.png hw compare
