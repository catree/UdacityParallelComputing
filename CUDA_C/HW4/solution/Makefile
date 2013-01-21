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

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

#no warnings otherwise thrust explodes output

NVCC_OPTS=-O3 -arch=sm_20 -m64

GCC_OPTS=-O3 -m64

student: compare main.o solution_func.o HW4.o loadSaveImage.o Makefile
	$(NVCC) -o hw main.o solution_func.o HW4.o loadSaveImage.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: ../student/main.cpp ../student/timer.h ../student/utils.h
	g++ -c ../student/main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

HW4.o: ../student/HW4.cu ../student/loadSaveImage.h ../student/utils.h
	$(NVCC) -c ../student/HW4.cu -I $(OPENCV_INCLUDEPATH) $(NVCC_OPTS)

loadSaveImage.o: ../student/loadSaveImage.cpp ../student/loadSaveImage.h
	g++ -c ../student/loadSaveImage.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

solution_func.o: solution_func.cu ../student/reference_calc.cpp ../student/utils.h
	$(NVCC) -c solution_func.cu $(NVCC_OPTS)

compare: ../student/compare.cpp ../student/utils.h Makefile
	g++ -o compare ../student/compare.cpp -I $(OPENCV_INCLUDEPATH) -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o *.png hw compare
