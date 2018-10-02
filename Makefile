CC=mpicc $(CFLAGS)
LD=mpicc $(LDFLAGS)
CFLAGS=-std=c89
LDFLAGS=
LIBS=-lm

HEADERS=src/nhbr_topo.h src/bench.h

.PHONY: all clean

all: nhbr-allgather

%.o: src/%.c $(HEADERS)
	$(CC) -c -o $@ $<

nhbr-allgather: nhbr_allgather.o nhbr_topo.o
	$(LD) -o $@ $^ $(LIBS)

clean:
	rm -f *.o
