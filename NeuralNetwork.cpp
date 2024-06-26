#include "NeuralNetwork.h"
#include <ctime>
#include <cassert>
#include <cmath>
#include <random>
#include <iostream>
//#include "debug.h"

#define VARIANCE_MULTIPLIER 100.0
#define STEP_COUNT 1000
#define DEBUGGING
#include "debug.h"


inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
//inline double dsigmoid(double x) { double sig = sigmoid(x); return sig * (1 - sig); }


int errors = 0;


void NeuralNetwork::init()
{
	this->neuron_count_prefix = std::vector<int>(get_layer_count()+1, 0);
	this->weight_count_prefix = std::vector<int>(get_layer_count(), 0);
	calculate_prefixes();
	this->neurons = std::vector<double>(neuron_count_prefix.back(), 0);
	this->weights = std::vector<double>(weight_count_prefix.back(), 0);
	this->biases  = std::vector<double>(neurons.size() - layer_sizes[0] - layer_sizes.back(), 0);
	this->neuron_derivatives = std::vector<double>(neurons.size());
	this->weight_derivatives = std::vector<double>(weights.size());
	this->bias_derivatives   = std::vector<double>(biases.size());
}

void NeuralNetwork::calculate_prefixes()
{
	for (int i = 1; i < get_layer_count(); i++)
	{
		neuron_count_prefix[i] = neuron_count_prefix[i-1] + layer_sizes[i-1];
		weight_count_prefix[i] = weight_count_prefix[i-1] + layer_sizes[i-1] * layer_sizes[i];
	}
	neuron_count_prefix[get_layer_count()] = neuron_count_prefix[get_layer_count()-1] + layer_sizes[get_layer_count()-1];
}


void NeuralNetwork::randomize_network()
{
	std::default_random_engine generator;
	generator.seed(time(NULL));
	for (int layer = 0; layer < get_layer_count() - 1; layer++)
	{
		std::normal_distribution<double> distribution(0, VARIANCE_MULTIPLIER / sqrt(static_cast<double>(layer_sizes[layer])));
		for (int j = 0; j < layer_sizes[layer+1]; j++)
		{
			for (int i = 0; i < layer_sizes[layer]; i++)
			{
				weights[get_weight_index(layer, i, j)] = distribution(generator);
			}
		}
	}
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) : layer_sizes(layer_sizes) 
{ 
	init(); 
	randomize_network();
}


NeuralNetwork::NeuralNetwork(std::istream& is)
{
	int layer_count;
	is >> layer_count;
	layer_sizes = std::vector<int>(layer_count);
	for (int& layer_size : layer_sizes)
		is >> layer_size;
	init();
	for (double& weight : weights)
		is >> weight;
	for (double& bias : biases)
		is >> bias;
}

void NeuralNetwork::write_network_values(std::ostream& os)
{
	os << layer_sizes.size() << "\n";
	for (int layer_size : layer_sizes)
		os << layer_size << "\n";
	for (double weight : weights)
		os << weight << "\n";
	for (double bias : biases)
		os << bias << "\n";
	std::cerr << errors << "\n";
}


const std::vector<double> NeuralNetwork::process(const std::vector<double>& values) 
{
	write_input(values);
	evaluate_values();
	return output_result();
}


void NeuralNetwork::write_input(const std::vector<double>& values)
{
	assert(values.size() == layer_sizes[0]);
	for (int i = 0; i < layer_sizes[0]; i++)
		neurons[get_neuron_index(0, i)] = values[i];
}

void NeuralNetwork::evaluate_values()
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

const std::vector<double> NeuralNetwork::output_result()
{
	std::vector<double> res(layer_sizes.back());
	for (int i = 0; i < layer_sizes.back(); i++)
		res[i] = neurons[get_neuron_index(get_layer_count()-1, i)];
	return res;
}

bool trained_before = false;

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& answers) 
{
	std::vector<double> gradient_sum = calculate_gradient_sum(inputs, answers);
	step_backwards(gradient_sum, 1);
	trained_before = true;
}

std::vector<double> NeuralNetwork::calculate_gradient_sum(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& answers)
{
	assert(inputs.size() == answers.size());
	std::vector<double> gradient_sum(get_gradient_size(), 0);
	for (int i = 0; i < inputs.size(); i++)
	{
		std::vector<double> current_gradient = back_propogate(inputs[i], answers[i]);
		assert(current_gradient.size() == gradient_sum.size());
		for (int j = 0; j < get_gradient_size(); j++)
			gradient_sum[j] += current_gradient[j];
	}
	return gradient_sum;
}

std::vector<double> NeuralNetwork::back_propogate(const std::vector<double>& input, const std::vector<double>& answer)
{
	write_input(input);
	evaluate_values();
	int mx = 0;
	for (int i = 1; i < layer_sizes.back(); i++)
		if (neurons[get_neuron_index(get_layer_count() - 1, i)] > neurons[get_neuron_index(get_layer_count() - 1, mx)])
			mx = i;
	if (answer[mx] < 0.5)
		errors++;
	evaluate_derivatives(answer);
	return get_gradient();
}

void NeuralNetwork::evaluate_derivatives(const std::vector<double>& answer)
{
	evaluate_output_neuron_derivatives(answer);
	for (int layer = get_layer_count() - 2; layer >= 1; layer--)
	{
		evaluate_weight_derivatives_in_layer(layer);
		evaluate_neuron_derivatives_in_layer(layer);
		evaluate_bias_derivatives_in_layer(layer);
	}
	evaluate_weight_derivatives_in_layer(0);
}

void NeuralNetwork::evaluate_output_neuron_derivatives(const std::vector<double>& answer)
{
	for (int i = 0; i < layer_sizes.back(); i++)
	{
		int neuron_index = get_neuron_index(get_layer_count() - 1, i);
		double diff = neurons[neuron_index] - answer[i];
		if (answer[i] > 0.1)
			diff *= 7;
		// cost_contribution is diff^2 so the derivative is 2 * diff * 1 with respect to ith neuron in the layer
		double derivative = 2 * diff * STEP_COUNT; // Making it bigger to make changes bigger
		//if (trained_before &&  abs(diff) > 0.1)
		//	debug(neurons[neuron_index], answer[i], derivative);
		neuron_derivatives[neuron_index] = derivative;
	}
}

void NeuralNetwork::evaluate_weight_derivatives_in_layer(int layer)
{
	for (int j = 0; j < layer_sizes[layer+1]; j++)
	{
		double derivative_multiplier = get_neuron_preactivation_derivative_multiplier(layer+1, j);
		for (int i = 0; i < layer_sizes[layer]; i++)
		{
			weight_derivatives[get_weight_index(layer, i, j)] = derivative_multiplier * neurons[get_neuron_index(layer, i)];
		}
	}
}

void NeuralNetwork::evaluate_neuron_derivatives_in_layer(int layer)
{
	for (int i = 0; i < layer_sizes[layer]; i++)
		neuron_derivatives[get_neuron_index(layer, i)] = 0;
	for (int j = 0; j < layer_sizes[layer+1]; j++)
	{
		double derivative_multiplier = get_neuron_preactivation_derivative_multiplier(layer+1, j);
		for (int i = 0; i < layer_sizes[layer]; i++)
		{
			neuron_derivatives[get_neuron_index(layer, i)] += derivative_multiplier * weights[get_weight_index(layer, i, j)];
		}
	}
}


void NeuralNetwork::evaluate_bias_derivatives_in_layer(int layer)
{
	for (int i = 0; i < layer_sizes[layer]; i++)
	{
		bias_derivatives[get_bias_index(layer, i)] = get_neuron_preactivation_derivative_multiplier(layer, i);
	}
}

double NeuralNetwork::get_neuron_preactivation_derivative_multiplier(int layer, int neuron)const
{
	double sig = neurons[get_neuron_index(layer, neuron)];
	double dsig = sig * (1 - sig);
	return neuron_derivatives[get_neuron_index(layer, neuron)] * dsig;
}

std::vector<double> NeuralNetwork::get_gradient()const
{
	std::vector<double> gradient(get_gradient_size());
	for (int i = 0; i < weight_derivatives.size(); i++)
	{
		gradient[i] = weight_derivatives[i];
	}
	for (int i = 0; i < bias_derivatives.size(); i++)
	{
		gradient[weight_derivatives.size() + i] = bias_derivatives[i];
	}
	debug(gradient);
	return gradient;
}


void NeuralNetwork::step_backwards(const std::vector<double>& gradient, double step_size)
{
	double total_step_size = 0;
	for (int i = 0; i < weights.size(); i++)
	{
		weights[i] -= gradient[i] * step_size;
		total_step_size += abs(gradient[i] * step_size);
	}
	for (int i = 0; i < biases.size(); i++)
	{
		biases[i] -= gradient[weights.size() + i] * step_size;
		total_step_size += abs(gradient[i] * step_size);
	}
	std::cerr << total_step_size << "\n";
}
