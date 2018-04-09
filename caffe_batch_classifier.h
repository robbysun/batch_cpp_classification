#ifndef CAFFE_BATCH_CLASSIFIER_H
#define CAFFE_BATCH_CLASSIFIER_H

using namespace caffe;  // NOLINT(build/namespaces)
//using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class BatchClassifier {
 public:
  BatchClassifier(
             const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file,
             const bool use_GPU,
             const int batch_size);

  std::vector<vector<Prediction>> Classify(const std::vector<cv::Mat> imgs, const size_t top_n);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const std::vector<cv::Mat> imgs);

  void WrapInputLayer(std::vector<std::vector<cv::Mat>> *input_batch);

  void Preprocess(const vector<cv::Mat> imgs,
                  std::vector<std::vector<cv::Mat>> *input_batch);

 private:
  std::shared_ptr<caffe::Net<float>> net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<std::string> labels_;
  int num_classes_;
  int batch_size_;
};

#endif  //CAFFE_BATCH_CLASSIFIER_H

