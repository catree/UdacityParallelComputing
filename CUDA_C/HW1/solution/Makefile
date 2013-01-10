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

solution: compare main.o solution_func.o
	$(NVCC) -o hw main.o solution_func.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)

main.o: ../student/main.cpp ../student/timer.h ../student/utils.h ../student/HW1.cpp Makefile
	g++ -c ../student/main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

solution_func.o: solution_func.cu ../student/reference_calc.cpp ../student/utils.h
	nvcc -c solution_func.cu $(NVCC_OPTS)

compare: ../student/compare.cpp ../student/utils.h Makefile
	g++ -o compare ../student/compare.cpp -I $(OPENCV_INCLUDEPATH) -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o *.png hw compare
