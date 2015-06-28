CC = g++
CFLAGS = -Wall -std=c++11 -O3 -fopenmp -I ../../libs/liblbfgs-1.10/include -I ../../libs/gzstream
LDFLAGS = -llbfgs -lgomp -lz -lstdc++ -L ../../libs/liblbfgs-1.10/lib/.libs -L ../../libs/gzstream

.PHONY : all clean
all: train

train: language.cpp language.hpp common.hpp ../../libs/liblbfgs-1.10/lib/.libs/liblbfgs.so ../../libs/gzstream/gzstream.o
	$(CC) $(CFLAGS) -o train language.cpp ../../libs/gzstream/gzstream.o $(LDFLAGS)

clean:
	rm train
