#pragma once

#include <string>
#include <fstream>
#include <unordered_map>
#include <sstream>

#include "NueralNetparams.hpp"

std::unordered_map<std::string, std::string> configfile_to_map(std::string &filename)
{
  std::unordered_map<std::string, std::string> configMap;
  std::ifstream configFile(filename);
  std::string line,key,value;
  while (std::getline(configFile, line))
  {
    std::istringstream iss(line);
    key.clear();
    if (std::getline(iss, key, '='))
    {
      std::erase(key, ' ');
      value.clear();
      if (std::getline(iss, value))
      {
        std::erase(value, ' ');
        configMap[key] = value;
      }
    }
  }
  return configMap;
}

NeuralNetworkParams loadConfig(std::string &filename)
{
  auto datamap{configfile_to_map(filename)};
  NeuralNetworkParams nn;
  nn.num_epochs = std::stoul(datamap.at("num_epochs"));
  nn.batch_size = std::stoul(datamap.at("batch_size"));
  nn.hidden_size = std::stoul(datamap.at("hidden_size"));
  nn.learning_rate = std::stod(datamap.at("learning_rate"));
  nn.rel_path_train_images = datamap.at("rel_path_train_images");
  nn.rel_path_train_labels = datamap.at("rel_path_train_labels");
  nn.rel_path_test_images = datamap.at("rel_path_test_images");
  nn.rel_path_test_labels = datamap.at("rel_path_test_labels");
  nn.rel_path_log_file = datamap.at("rel_path_log_file");
  return nn;
}