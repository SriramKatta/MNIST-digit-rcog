#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <iomanip>

#include "configreader.hpp"
#include "Neuralnet.hpp"

size_t inverse_one_hot(const auto& label);


int main(int argc, char const *argv[])
{
  bool debugprint{false};
  bool accurprint{false};
  if (argc == 1){
    debugprint = true;
    accurprint = true;
    argv[1] = "mnist-configs/input.config";
  }
  std::string configpath(argv[1]);
  NeuralNetworkParams nnparams{loadConfig(configpath)};
  Neuralnet model(784, 10, nnparams);
  mnistDatareader training_data(nnparams, true);
  auto no_images{training_data.get_item_count()};
  auto no_batches{no_images / nnparams.batch_size};

  model.train();

  size_t batchstart{};

  Eigen::MatrixXd imgs, labels,loss;
  for (size_t epochnum = 0; epochnum < nnparams.num_epochs; ++epochnum)
  {
    Eigen::MatrixXd imgs, labels;
    for (size_t batchno = 0; batchno < no_batches; ++batchno)
    {
      batchstart = batchno * nnparams.batch_size;
      imgs = training_data.getimage(batchstart, nnparams.batch_size);
      labels = training_data.getlabel(batchstart, nnparams.batch_size);
      loss = model(imgs, labels);
      if (debugprint && batchstart % 10000 == 0)
      {
        auto avgloss = loss.sum() / nnparams.batch_size;
        std::cout << "epoch is : " << std::setw(1) << epochnum
                  << " | 10000 th batch start is: " << std::setw(2) << batchstart/1000
                  << " | avgloss : "<< std::fixed << std::setw(6) << std::setfill('0')<< std::setprecision(3) << avgloss << std::endl;
      }
    }
    //training_data.shuffle();
  }

  mnistDatareader testing_data(nnparams, false);
  size_t correct{0}, total{testing_data.get_item_count()};
  size_t predicted{}, actual{};

  std::ofstream logfile(nnparams.rel_path_log_file);

  auto total_batches{total / nnparams.batch_size};

  model.eval();

  Eigen::MatrixXd outputs;

  #pragma opm parallel for
  for (size_t batchno = 0; batchno < total_batches; ++batchno)
  {
    batchstart = batchno * nnparams.batch_size;
    logfile << "Current batch: " << batchno << std::endl;
    imgs = testing_data.getimage(batchstart, nnparams.batch_size);
    labels = testing_data.getlabel(batchstart, nnparams.batch_size);
    outputs = model(imgs);
    for (size_t i = 0; i < nnparams.batch_size; ++i)
    {
      predicted = inverse_one_hot(outputs.col(i));
      actual = inverse_one_hot(labels.col(i));
      logfile << " - image " << i + batchstart
              << ": Prediction=" << predicted
              << ". Label=" << actual << std::endl;
      if (predicted == actual)
        ++correct;
    }
  }
  if(accurprint)
    std::cout << "accuracy on test data is " << 100 * static_cast<double>(correct) / total << "\n";
  return 0;
}

size_t inverse_one_hot(const auto& label){
  return std::distance(label.begin(), std::max_element(label.begin(), label.end()));
}
