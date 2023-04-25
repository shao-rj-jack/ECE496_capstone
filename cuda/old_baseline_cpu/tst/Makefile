CC=nvcc
ODIR=build
LDIR=lib
IDIR=h
SDIR=src
FLAGS=-I$(IDIR)
LIBS=
DEBUG=

_DEPS=cuda_n.h
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ=cuda.o main.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS) $(DEBUG)

main: $(OBJ)
	$(CC) -o $@ $^ $(FLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(IDIR)/.*.swp $(SDIR)/.*.swp core main 

