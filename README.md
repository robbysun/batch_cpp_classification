Title: CaffeNet C++ Batch Classification

Description: 
This code performs image batch classification using the low-level C++ API. The code was 
copied from erogol's CaffeBatchPrediction gist (https://gist.github.com/erogol/67e02e87f94ce9dc0c63), 
which in turn was based on Caffe's cpp_classification example.

I made a few minor changes, mainly the class/member names. I also added a main function 
to make this project a stand alone application.

# Classifying ImageNet: using the C++ API

Caffe, at its core, is written in C++. It is possible to use the C++
API of Caffe to implement an image classification application similar
to the Python code presented in one of the Notebook examples. To look
at a more general-purpose example of the Caffe C++ API, you should
study the source code of the command line tool `caffe` in `tools/caffe.cpp`.

## Presentation

A simple C++ code is proposed in
`examples/cpp_classification/classification.cpp`. For the sake of
simplicity, that example does not support oversampling of a single
sample nor batching of multiple independent samples. That example is
not trying to reach the maximum possible classification throughput on
a system, but special care was given to avoid unnecessary
pessimization while keeping the code readable.

This application tries to improve the classification throughput. Compared 
with the original 'cpp_classification', the speedup can be 5X or more, 
depending on the hardware.

## Compiling

Use the Makefile in the directory. You may need to change the Caffe and/or 
OpenCV paths to reflect your installation configurations.

## Usage

To use the pre-trained CaffeNet model with the classification example,
you need to download it from the "Model Zoo" using the following
script:
```
./scripts/download_model_binary.py models/bvlc_reference_caffenet
```
The ImageNet labels file (also called the *synset file*) is also
required in order to map a prediction to the name of the class:
```
./data/ilsvrc12/get_ilsvrc_aux.sh
```
Using the files that were downloaded, we can classify all the images in 
a directory using this command:
```
./caffe_batch_classifier \
  models/bvlc_reference_caffenet/deploy.prototxt \
  models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  data/ilsvrc12/imagenet_mean.binaryproto \
  data/ilsvrc12/synset_words.txt \
  directory_of_test_images
```

Optionally, you can specify the batch size and the top-n classes for the 
output. For example, the following command:
```
./caffe_batch_classifier \
  models/bvlc_reference_caffenet/deploy.prototxt \
  models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  data/ilsvrc12/imagenet_mean.binaryproto \
  data/ilsvrc12/synset_words.txt \
  dir_for_your_test_images \
  32 \
  10
```
sets the batch size to be 32, and outputs the top-10 classes for each 
image. The default batch size is 4, and top-n is 5, like in the 
original 'cpp_classification' example.

## Improving Performance

To further improve performance, you will need to leverage the GPU
more, here are some guidelines:

* Use multiple classification threads to ensure the GPU is always fully
utilized and not waiting for an I/O blocked CPU thread.
