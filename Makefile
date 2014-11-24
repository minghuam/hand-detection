CC = g++
CXXFLAGS += -I. 
LDFLAGS = $(shell pkg-config opencv --libs)$

SRC = test.cpp handdetector.cpp \
	./HandDetector/Classifier.cpp \
	./HandDetector/FeatureComputer.cpp \
	./HandDetector/LcBasic.cpp

OBJS = $(SRC:.cpp=.o)
.cpp.o:
	$(CC) $(CXXFLAGS) -c $< -o $@

all: $(OBJS) $(SRC)
	$(CC) $(OBJS) $(LDFLAGS) -o test

clean:
	rm test *.o ./HandDetector/*.o
