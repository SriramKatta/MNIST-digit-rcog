#pragma once

#include <string>

struct NeuralNetworkParams
{
  size_t num_epochs{};
  size_t batch_size{};
  size_t hidden_size{};
  double learning_rate{};
  std::string rel_path_train_images{};
  std::string rel_path_train_labels{};
  std::string rel_path_test_images{};
  std::string rel_path_test_labels{};
  std::string rel_path_log_file{};
};