.PHONY: all clean

CC=g++
CFLAGS+= -std=c++11 -O2
LDFLAGS+=-lm     # link to math library

TARGET=train test

all: $(TARGET)
# type make/make all to compile test_hmm

train: train.cpp
	$(CC) $(CFLAGS) train.cpp -o train

test: test.cpp
	$(CC) $(CFLAGS) test.cpp -o test

clean:
	$(RM) $(TARGET) # type make clean to remove the compiled file