# Makefile for python+novas 3.1 routines.
# The idea is to compile and archive them into libnovas.a
# The version compiled with FORTRAN routines is appfor
#
# The ephemeris file is called JPLEPH based on binEphem.405
#
# Eventually you can create a wrapper for Python with swig
# and use the module novas with Python
# use swig-1.3.x or higher

OS_NAME = $(shell uname -s)

ifeq ($(OS_NAME),Darwin)
  INCLUDE=/usr/include/python2.7
  LIBRARY=/usr/lib/python2.7
  CFLAGS= -O2 -fPIC -m64 -arch x86_64
  CC = gcc
  FORTRAN = g77
endif

ifeq ($(OS_NAME),Linux)
  INCLUDE=/usr/include/python2.7
  LIBRARY=/usr/lib/python2.7
  CFLAGS= -O2 -I/usr/include/cfitsio -L/usr/local/lib
  CC = gcc
  FORTRAN = g77
endif

OBJS =	\
	novascon.o \
	readeph0.o \
	solsys1.o  \
	nutation.o \
	eph_manager.o \
    uvfits.o \
    slalib.o \
	example.o \
    csrhrawdata.o \
    csrhuvfits.o \
	novas.o \


all:	app csrhuvfits example  libnovas.a
#_novas.so

libnovas.a:	$(OBJS)
	ar rv $@ $?

example:	example.c libnovas.a
	$(CC) $(CFLAGS) -o example example.c libnovas.a -lm

csrhuvfits:	csrhuvfits.c libnovas.a
	$(CC) $(CFLAGS) -o csrhuvfits csrhuvfits.c libnovas.a -L/usr/local/lib -lm -lcfitsio

app:	main.c libnovas.a
	$(CC) $(CFLAGS) -o app main.c libnovas.a -lm

appfor:	main.c jplsubs.o jplint.o
	$(CC) -o appfor main.c novas.o novascon.o readeph0.o solsys1.c jplsubs.o jplint.o  -lm -lg3c

jplsubs.o: jplsubs.f
	$(FORTRAN) $(CFLAGS) -c jplsubs.f

jplint.o:  jplint.f
	$(FORTRAN) $(CFLAGS) -c jplint.f

#novas_wrap.c:	novas.i
#	swig -python  $<

#novas_wrap.o:	novas_wrap.c
#	$(CC) $(CFLAGS) -I$(INCLUDE) -c $<

#_novas.so:	novas_wrap.o libnovas.a
#	$(CC) $(CFLAGS) -L/usr/lib/python2.7 -lpython -lm -shared -o _novas.so novas_wrap.o novas.o novascon.o readeph0.o solsys1.o nutation.o eph_manager.o

clean:
	rm -f *.o libnovas.a *.so
