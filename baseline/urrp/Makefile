CC = g++
CFLAGS = -Wall -std=c++11 -O3 -fopenmp -I ../../libs/gzstream
SCFLAGS = -Wall -static -std=c++11 -O3 -fopenmp -I ../../libs/gzstream
LDFLAGS = -lgomp -lz -lstdc++ -L ../../libs/gzstream

.PHONY : all clean
all: urrp

urrp: main.cpp urrp.cpp urrp.h utils.cpp utils.h ../../libs/gzstream/gzstream.o
	$(CC) $(SCFLAGS) main.cpp utils.cpp urrp.cpp ../../libs/gzstream/gzstream.o -o urrp $(LDFLAGS)

durrp: main.cpp urrp.cpp urrp.h utils.cpp utils.h ../../libs/gzstream/gzstream.o
	$(CC) $(CFLAGS) main.cpp utils.cpp urrp.cpp ../../libs/gzstream/gzstream.o -o urrp $(LDFLAGS)

clean:
	rm urrp
