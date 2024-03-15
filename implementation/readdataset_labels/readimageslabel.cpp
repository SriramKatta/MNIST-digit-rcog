#include <iostream>
#include <string>
#include <Eigen/Dense>

#include "mnistdatareader.hpp"

void write_to_file(const std::string& outfile, const Eigen::MatrixXd& vec){
  Eigen::IOFormat prettyprint(Eigen::StreamPrecision,Eigen::DontAlignCols,"\n", "\n");
  std::ofstream out_file(outfile);
  out_file << 1 << "\n";
  out_file << vec.size() << "\n";
  out_file << vec.format(prettyprint) << "\n";
};

int main(int argc, char const *argv[])
{
  if (argc != 4){
    argv[1] = "mnist-datasets/train-labels.idx1-ubyte";
    argv[2] = "label_out.txt";
    argv[3] = "0";
  }
  std::string label_dataset_path(argv[1]);
  std::string image_tensor_label_out_path(argv[2]);
  size_t label_index=  std::stoull(argv[3]);

  mnistLabelReader labelreader(label_dataset_path);

  write_to_file(image_tensor_label_out_path, labelreader.getlabel(label_index));

  return 0;
}
