CXX=icpc
CC=$(CXX)
NVCC=nvcc

#  -Ofast -march=native -pipe -funroll-loops
CFLAGS= -std=c++11 -O2 -ipo -I/usr/local/cuda-7.5/targets/x86_64-linux/include/ -march=native -g
CXXFLAGS=$(CFLAGS) 

NVFLAGS= -arch=sm_50 -O2 -ccbin=g++ -g

LDFLAGS= -L/usr/local/cuda-7.5/lib64 
LDLIBS=-lcuda -lcudart -lcublas -lcusolver -lcurand


CUOBJS= gpu.o
OBJS= main.o 

all:main

main:$(CUOBJS) $(OBJS)


%.o:%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CUOBJS) $(OBJS) $(EXE)

redo: clean all
