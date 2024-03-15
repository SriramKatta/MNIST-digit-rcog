#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <concepts>
#include <bit>
#include <climits>

class binaryDatareader
{
public:
  binaryDatareader() = delete;
  binaryDatareader(binaryDatareader &) = delete;
  binaryDatareader(binaryDatareader &&) = delete;
  binaryDatareader &operator=(binaryDatareader &) = delete;
  binaryDatareader &operator=(binaryDatareader &&) = delete;

  explicit binaryDatareader(std::string &filepath)
      : datafile(filepath, std::ios::binary) {}

  void skip(size_t no_of_bytes)
  {
    datafile.seekg(no_of_bytes, std::ios_base::cur);
  }

  template <std::integral datatype>
  datatype get()
  {
    datatype data{};
    datafile.read(std::bit_cast<char *>(&data), sizeof(data));
    data = swap_endian(data);
    return data;
  }

  ~binaryDatareader() = default;

private:
  template <typename T>
  T swap_endian(T u)
  {
    static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

    union
    {
      T u;
      unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
      dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
  }
  std::ifstream datafile;
};