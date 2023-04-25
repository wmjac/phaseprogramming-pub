include Makefile.inc # Define the Python and numpy include flags here (may be system-specific).

CC = gcc
CYTHON = cython
CPPFLAGS = $(PYTHONINC) $(NUMPYINC)
CFLAGS = -fPIC -O3 -g -Wall
PWD := `pwd`
LDFLAGS = -L$(PWD) -Wl,-rpath=`pwd`
LIBS = -lm

all:	cfh.so

cfh.so:	cfh.o
	$(CC) -shared -o cfh.so cfh.o $(LDFLAGS)

cfh.c:	cfh.pyx
	$(CYTHON) cfh.pyx

-include $(OBJS:.o=.d)
-include cfh.d

%.o: %.c
	$(CC) $(CPPFLAGS) -c $(CFLAGS) $*.c -o $*.o
	$(CC) $(CPPFLAGS) -MM $*.c > $*.d

.PHONY : clean
clean:
	-rm *.so *.o *.d cfh.c
