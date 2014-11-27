CC = g++
CFLAGS = -std=c++0x
INCLUDES = -I./
LFLAGS = $(shell pkg-config opencv --libs)$
LIBS = 

TARGET = test

SRCS = test.cpp image_cache.cpp feature_extractor.cpp feature.cpp color_feature.cpp handdetector.cpp LcBasic.cpp Classifier.cpp FeatureComputer.cpp

OBJS = $(SRCS:.cpp=.o)

.PHONY: depend clean

all: $(TARGET)
	@echo  done!

$(TARGET): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(OBJS) $(LFLAGS) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(TARGET)