#pragma once
#include <istream>
#include <ostream>
#include <vector>

class NeuralNetwork
{
	std::vector<int> layer_sizes;
	std::vector<int> neuron_count_prefix;
	std::vector<int> weight_count_prefix;
	std::vector<double> weights;
	std::vector<double> neurons;
	std::vector<double> biases;
	std::vector<double> neuron_derivatives;
	std::vector<double> weight_derivatives;
	std::vector<double> bias_derivatives;
public:
	const std::vector<double> process(const std::vector<double>& values);
	void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& answers);
	void write_network_values(std::ostream& os);	
	NeuralNetwork(const std::vector<int>& layer_sizes); 
	NeuralNetwork(std::istream& is); 
private:
	void init();
	void calculate_prefixes();
	void randomize_network();
	void write_input(const std::vector<double>& values);
	void evaluate_values();
	const std::vector<double> output_result();
	std::vector<double> calculate_gradient_sum(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& answers);
	std::vector<double> back_propogate(const std::vector<double>& input, const std::vector<double>& answer);
	void evaluate_derivatives(const std::vector<double>& answer);
	void evaluate_output_neuron_derivatives(const std::vector<double>& answer);
	void evaluate_weight_derivatives_in_layer(int layer);
	void evaluate_neuron_derivatives_in_layer(int layer);
	void evaluate_bias_derivatives_in_layer(int layer);
	double get_neuron_preactivation_derivative_multiplier(int layer, int neuron)const;
	std::vector<double> get_gradient()const;
	void step_backwards(const std::vector<double>& gradient, double step_size);
	inline int get_neuron_index(int layer, int pos) const { return neuron_count_prefix[layer] + pos; }
	inline int get_weight_index(int layer, int left_pos, int right_pos) { return weight_count_prefix[layer] + right_pos * layer_sizes[layer] + left_pos; }
	inline int get_layer_count() const { return layer_sizes.size(); }
	inline int get_bias_index(int layer, int pos) const { return pos + neuron_count_prefix[layer] - layer_sizes[0]; }
	inline int get_gradient_size() const { return weights.size() + biases.size(); }
};
