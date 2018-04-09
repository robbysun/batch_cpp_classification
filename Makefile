CC = g++
INCLUDE = -I$(CAFFE_ROOT)/include -I$(CAFFE_ROOT)/build/src -I/usr/local/cuda/include
CFLAGS = -O3 -Wall $(INCLUDE) -DUSE_OPENCV -std=c++11
SRCS = caffe_batch_classifier.cpp batch_classifier_test.cpp
PROG = caffe_batch_classifier

CAFFE_LIB = -L$(CAFFE_ROOT)/build/lib -lcaffe -lboost_system -lglog
OPENCV_LIB = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV_LIB) $(CAFFE_LIB)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean:
	rm -f $(PROG)
	rm -f $(PROG).o
	rm -f *~
