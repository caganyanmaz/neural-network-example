#include <iostream>
#include <ctime>
#include <random>
#include <algorithm>
#include <cassert>
#include <fstream>
#include "NeuralNetwork.h"
#include "MNISTReader.h"

std::string s = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/()1{}[]?-_+~<>i!lI;:,^`'.";
const int SIZE = 28;
const int BATCH_SIZE = 100;
const int DIGIT_COUNT = 10;
const int BATCH_COUNT = 100;
const std::vector<int> default_layer_sizes = {784, 100, 100, 10};



//#define PRINT_EXAMPLE


void process(NeuralNetwork& nw)
{
	int number_of_images, image_size;
	int number_of_labels;
	uchar** images = read_mnist_images("data/train-images.idx3-ubyte", number_of_images, image_size);
	uchar* labels  = read_mnist_labels("data/train-labels.idx1-ubyte", number_of_labels);
	assert(number_of_images == number_of_labels);
	std::cout << image_size << "\n";
#ifdef PRINT_EXAMPLE
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			for (int k = 0; k < SIZE; k++)
				std::cout << s[images[i][j*SIZE+k] / 4];
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	for (int i = 0; i < 10; i++)
		std::cout << (int)labels[i] << " ";
	std::cout << "\n";
#else
	for (int batch_count = 0; batch_count < BATCH_COUNT; batch_count++)
	{
		std::vector<std::vector<double>> values, answers;
		for (int i = 0; i < BATCH_SIZE; i++)
		{
			//int random_data = rand() % number_of_images;
			values.push_back(std::vector<double>(image_size));
			answers.push_back(std::vector<double>(10));
			for (int j = 0; j < image_size; j++)
			{
				values.back()[j] = static_cast<double>(images[i][j]) / sizeof(uchar);
			}
			int answer = labels[i];
			for (int j = 0; j < DIGIT_COUNT; j++)
			{
				answers.back()[j] = (answer == j) ? 1.0 : 0.0;
			}
		}
		nw.train(values, answers);
		std::cout << "Trained with " << batch_count << " batches...\n";
	}
	std::ofstream os("weight-data.txt");
	nw.write_network_values(os);
#endif
	delete images;
	delete labels;
}


int main(int argc, char **argv)
{
	srand(time(NULL));
	std::reverse(s.begin(), s.end());
	if (argc == 1)
	{
		// Initialize a new neural network
		NeuralNetwork nw(default_layer_sizes);
		process(nw);
	}
	else
	{
		std::cout << "Loading file: " << argv[1] << "\n";
		std::string file_name(argv[1]);
		std::ifstream in(file_name);
		NeuralNetwork nw(in);
		process(nw);
	}
}

