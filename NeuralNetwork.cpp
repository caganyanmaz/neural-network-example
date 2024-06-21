#include "NeuralNetwork.h"
#include <cmath>
#include <random>

#define LOWER_BOUND -10
#define UPPER_BOUND 10
#define MOVEMENT 100


inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
inline double dsigmoid(double x) { double sig = sigmoid(x); return sig * (1 - sig); }


void NeuralNetwork::init()
{
	this->neuron_count_prefix = std::vector<int>(get_layer_count()+1, 0);
	this->weight_count_prefix = std::vector<int>(get_layer_count(), 0);
	for (int i = 1; i < get_layer_count(); i++)
	{
		neuron_count_prefix[i] = neuron_count_prefix[i-1] + layer_sizes[i-1];
		weight_count_prefix[i] = weight_count_prefix[i-1] + layer_sizes[i-1] * layer_sizes[i];
	}
	neuron_count_prefix[get_layer_count()] = neuron_count_prefix[get_layer_count()-1] + layer_sizes[get_layer_count()-1];
	this->neurons = std::vector<double>(neuron_count_prefix.back(), 0);
	this->weights = std::vector<double>(weight_count_prefix.back(), 0);
	this->biases  = std::vector<double>(neurons.size() - layer_sizes[0] - layer_sizes.back(), 0);
	this->neuron_derivatives = std::vector<double>(neurons.size());
	this->weight_derivatives = std::vector<double>(weights.size());
	this->bias_derivatives   = std::vector<double>(biases.size());
	std::uniform_real_distribution<double> unif(LOWER_BOUND, UPPER_BOUND);
	std::default_random_engine re;
	for (double& val : weights)
		val = unif(re);
	for (double& val : biases)
		val = unif(re);

}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) : layer_sizes(layer_sizes) { init(); }

const std::vector<double>& NeuralNetwork::process(const std::vector<double>& values) { return biases; }


void write_input(const std::vector<double>& values)
{
	assert(values.size() == layer_sizes[0]);
	for (int i = 0; i < layer_sizes[0]; i++)
		neurons[get_neuron_index(0, i)] = values[i];
}

void NeuralNetwork::evaluate(const std::vector<double>& values)
{
	for (int layer = 1; layer < get_layer_count(); layer++)
	{
		for (int i = 0; i < layer_sizes[layer]; i++)
		{
			double res = biases[get_bias_index(layer, i)];
			for (int j = 0; j < layer_sizes[layer-1]; j++)
				res += weights[get_weight_index(layer-1, j, i)] * neurons[get_neuron_index(layer-1, j)];
			neurons[get_neuron_index(layer, i)] = sigmoid(res);
		}
	}
}

const std::vector<double> output_result()
{
	std::vector<double> res = layer_sizes.back();
	for (int i = 0; i < layer_sizes.back(); i++)
		res[i] = neurons[get_neuron_index(get_layer_count()-1, i)];
	return res;
}

void NeuralNetwork::back_propogate(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& answers) {}
