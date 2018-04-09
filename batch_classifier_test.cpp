#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace cv::dnn;

#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe_batch_classifier.h"


void print_usage(const char *argv[])
{
    std::cout << "Usage: " << argv[0] << " deploy_file weight_file image_mean_file label_file image_folder batch_size top_n" << std::endl;
    std::cout << "where batch_size (default 4) and top_n (default 5) are optional." << std::endl;
    std::cout << "Note top_n must be set after batch_size." << std::endl;
}

int main(const int argc, const char *argv[]) 
{
  if (argc < 6) {
    print_usage(argv);
    return -1;
  }

  // Command-line options
  std::string model_file   = string(argv[1]);
  std::string trained_file = string(argv[2]);
  std::string mean_file    = string(argv[3]);
  std::string label_file   = string(argv[4]);
  std::string folder       = string(argv[5]);

  size_t num_batch_imgs;
  size_t top_n;
  switch (argc) {
    case 6:
      num_batch_imgs = 4;
      top_n = 5;
      break;
    case 7:
      num_batch_imgs = stoi(string(argv[6]));
      top_n = 5;
      break;
    case 8:
      num_batch_imgs = stoi(string(argv[6]));
      top_n = stoi(string(argv[7]));
      break;
    default:
      print_usage(argv);
      return -1;
  }


  // Note we are using the Opencv's embedded "String" class
  std::vector<String> all_imgs;
  glob(folder, all_imgs);

  //--------------------------------------------------------
  // Loop through all image files in the folder
  //--------------------------------------------------------
  std::vector<cv::Mat> tst_imgs;
  std::vector<std::string> fnames;
  for (size_t i = 0; i < all_imgs.size(); ++i) {
    String infile = all_imgs[i];
    cv::Mat inimg = imread(infile);
    if (inimg.empty()) {
      std::cerr << "Can't read image from the file: " << infile << std::endl;
      continue;
    }
    else {
      tst_imgs.push_back(inimg);
      fnames.push_back(string(infile));
    }
  }
  const size_t num_total_imgs = tst_imgs.size();
  std::cout << "Total test images: " << num_total_imgs << std::endl << std::endl;

  size_t num_tested_imgs = 0;

  // Instanciate the class
  BatchClassifier bclassifier(model_file, trained_file, mean_file, label_file, true, num_batch_imgs);

  while (num_tested_imgs < num_total_imgs)
  {
    if (num_batch_imgs > (num_total_imgs-num_tested_imgs)) num_batch_imgs = num_total_imgs-num_tested_imgs;

    // Get the test images for the current batch
    std::vector<cv::Mat>::const_iterator first = tst_imgs.begin() + num_tested_imgs;
    std::vector<cv::Mat>::const_iterator last = tst_imgs.begin() + num_tested_imgs + +num_batch_imgs;
    std::vector<cv::Mat> batch_imgs(first, last);

    std::vector<std::vector<Prediction>> predictions = bclassifier.Classify(batch_imgs, top_n);

    /* Print the top N predictions. */
    for (size_t j = 0; j < predictions.size(); ++j) {
      std::vector<Prediction> p = predictions[j];
      size_t top_n = p.size();
      std::cout << "----- Classification for " << fnames[num_tested_imgs+j] << "-----" << std::endl;
      for (size_t i = 0; i < top_n; ++i) {
        std::cout << std::fixed << std::setprecision(9) << p[i].second << " - \""
              << p[i].first << "\"" << std::endl;
      }
    }

    // Keep counting of number of tested images
    num_tested_imgs += num_batch_imgs;
  }

  std::cout << "Batch classification has completed." << std::endl;
  return 0;
}

