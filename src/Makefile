CC = g++
CFLAGS = -Wall -static -std=c++11 -O3 -fopenmp -I ../libs/gzstream
LDFLAGS = -lgsl -lm -lgslcblas -lgomp -lgzstream -lz -lstdc++\
					-L ../libs/gzstream

LSOURCE = json2txt.cpp utils.cpp
LHEADER = utils.h

.PHONY: all clean
all: json2txt

json2txt: $(LSOURCE) $(LHEADER) ../libs/gzstream/gzstream.o
	$(CC) $(CFLAGS) $(LSOURCE) -o json2txt $(LDFLAGS)

clean:
	rm json2txt
