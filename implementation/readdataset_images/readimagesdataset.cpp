#include <iostream>
#include <string>
#include <Eigen/Dense>

#include "mnistdatareader.hpp"

void write_to_file(const std::string& outfile, const Eigen::MatrixXd& mat){
  Eigen::IOFormat prettyprint(Eigen::StreamPrecision,Eigen::DontAlignCols,"\n", "\n");
  std::ofstream out_file(outfile);
  out_file << 2 << "\n";
  out_file << floor(sqrt(mat.size())) << "\n";
  out_file << floor(sqrt(mat.size())) << "\n";
  out_file << mat.format(prettyprint) << "\n";
}

int main(int argc, char const *argv[]){
    if (argc != 4){
      argv[1] = "mnist-datasets/train-images.idx3-ubyte";
      argv[2] = "image_out.txt";
      argv[3] = "0";
    }
    std::string image_dataset_path(argv[1]);
    std::string image_tensor_out_path(argv[2]);
    size_t image_index=  std::stoull(argv[3]);

    mnistImageReader imagereader(image_dataset_path);

    write_to_file(image_tensor_out_path, imagereader.getimage(image_index));
    
    return 0;
}

