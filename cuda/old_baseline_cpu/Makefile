CC=g++
ODIR=build
LDIR=lib
IDIR=h
SDIR=src
FLAGS=-I$(IDIR)
LIBS=
DEBUG=-g

_DEPS=Tensor.hpp convolution.hpp
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ=main.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)
	mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(FLAGS) $(DEBUG)


main: $(OBJ)
	$(CC) -o $@ $^ $(FLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -rf $(ODIR) $(IDIR)/.*.swp $(SDIR)/.*.swp core main 


