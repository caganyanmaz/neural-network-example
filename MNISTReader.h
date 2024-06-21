#pragma once
#include <string>
typedef unsigned char uchar;
uchar** read_mnist_images(std::string full_path, int& number_of_images, int& image_size);
uchar* read_mnist_labels(std::string full_path, int& number_of_labels);
