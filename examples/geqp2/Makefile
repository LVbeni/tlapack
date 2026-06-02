-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: geqp2_colpiv

geqp2_colpiv: geqp2_colpiv.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f geqp2_colpiv