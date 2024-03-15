#pragma once

#include <string>
#include <Eigen/Dense>
#include <random>

#include "bindatareader.hpp"
#include "NueralNetparams.hpp"

uint32_t get_no_of_items(std::string &image_dataset_path)
{
  binaryDatareader dataset(image_dataset_path);
  dataset.get<uint32_t>();
  return dataset.get<uint32_t>();
}

class mnistImageReader{
  using eigenBlock = Eigen::Block<Eigen::MatrixXd>;
public:
  mnistImageReader() = delete;
  mnistImageReader(mnistImageReader &) = delete;
  mnistImageReader(mnistImageReader &&) = delete;
  mnistImageReader &operator=(mnistImageReader &) = delete;
  mnistImageReader &operator=(mnistImageReader &&) = delete;
  mnistImageReader(std::string& path) : image_dataset_path(path){
    readimages();
  }
  eigenBlock getimage(size_t image_index){
    return images_with_padding.block(0,image_index,imagesize,1);
  }

  eigenBlock getimage(size_t start_index, size_t noofimages){
    return images_with_padding.block(0,start_index,imagesize,noofimages);
  }

protected:
  void readimages(){
    binaryDatareader dataset(image_dataset_path);
    auto magic_num{dataset.get<uint32_t>()};
    auto no_of_items{dataset.get<uint32_t>()};
    auto no_rows{dataset.get<uint32_t>()};
    auto no_cols{dataset.get<uint32_t>()};
    this-> imagesize = no_rows * no_cols;
    images_with_padding.resize(imagesize,no_of_items);
    Eigen::VectorXd fillvec(imagesize);
    for (auto colref : images_with_padding.colwise())
    {
      for (size_t i = 0; i < imagesize; ++i)
        fillvec(i) = static_cast<double>(dataset.get<uint8_t>()) / 255.0;
      colref = fillvec;
    }
  }
  size_t imagesize;
  std::string image_dataset_path;
  Eigen::MatrixXd images_with_padding;

};


class mnistLabelReader{
  using eigenBlock = Eigen::Block<Eigen::MatrixXd>;
public:
  mnistLabelReader() = delete;
  mnistLabelReader(mnistLabelReader &) = delete;
  mnistLabelReader(mnistLabelReader &&) = delete;
  mnistLabelReader &operator=(mnistLabelReader &) = delete;
  mnistLabelReader &operator=(mnistLabelReader &&) = delete;
  mnistLabelReader(std::string& path):label_dataset_path(path){
    readlabels();
  }

  eigenBlock getlabel(size_t image_index){
    return labels.block(0,image_index, 10,1);
  }

  eigenBlock getlabel(size_t start_index, size_t nooflables){
    return labels.block(0,start_index,10,nooflables);
  }

protected:
  void readlabels(){
    binaryDatareader dataset(label_dataset_path);
    auto magic_num{dataset.get<uint32_t>()};
    auto no_of_items{dataset.get<uint32_t>()};
    labels = Eigen::MatrixXd::Zero(10,no_of_items);
    for(auto colref : labels.colwise())
      colref(dataset.get<uint8_t>()) = 1.0;
  }
  std::string label_dataset_path;
  Eigen::MatrixXd labels;
  
};


class mnistDatareader:public mnistImageReader, public mnistLabelReader{
public:
  mnistDatareader(NeuralNetworkParams& nn, bool train):
  mnistImageReader(image_path_selector(nn,train)),
  mnistLabelReader(label_path_selector(nn,train)),
  no_of_items(get_no_of_items(label_path_selector(nn,train))){}
  size_t get_item_count(){
    return no_of_items;
  }

  void shuffle(){
    if(images_with_padding.size() == 1)
      return;
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::vector<std::size_t> indices(images_with_padding.cols());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    for (std::size_t i = 0; i < indices.size(); ++i) {
      images_with_padding.col(i).swap(images_with_padding.col(indices[i]));
      labels.col(i).swap(labels.col(indices[i]));
    }
  }

private:
  std::string& image_path_selector(NeuralNetworkParams& nn, bool train){
    return train?nn.rel_path_train_images:nn.rel_path_test_images;
  }
  std::string& label_path_selector(NeuralNetworkParams& nn,bool train){
    return train?nn.rel_path_train_labels:nn.rel_path_test_labels;
  }
  size_t no_of_items;
};
