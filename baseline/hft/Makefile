CC = g++
CFLAGS = -Wall -O3 -fopenmp -Iliblbfgs-1.10/include -Igzstream
LDFLAGS = -llbfgs -lgomp -lgzstream -lz -lstdc++ -Lliblbfgs-1.10/lib/.libs -Lgzstream

all: train

liblbfgs-1.10/lib/.libs/liblbfgs.so:
	cd liblbfgs-1.10 && ./configure && make

gzstream/gzstream.o:
	cd gzstream && make

train: language.cpp language.hpp common.hpp liblbfgs-1.10/lib/.libs/liblbfgs.so gzstream/gzstream.o
	$(CC) $(CFLAGS) -o train language.cpp gzstream/gzstream.o $(LDFLAGS)

clean:
	cd liblbfgs-1.10 && make clean
	cd gzstream && make clean
	rm train
