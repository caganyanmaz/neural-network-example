#pragma once
#include <vector>

class NeuralNetwork
{
	const std::vector<int> layer_sizes;
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
	void back_propogate(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& answers);
	NeuralNetwork(const std::vector<int>& layer_sizes); //: layer_sizes(layer_sizes) { init(); }
private:
	void init();
	void evaluate();
	void write_input(const std::vector<double>& values);
	const std::vector<double> output_result();
	inline int get_neuron_index(int layer, int pos) const { return ( (layer == 0) ? 0 : neuron_count_prefix[layer-1]) + pos; }
	inline int get_weight_index(int layer, int left_pos, int right_pos) { return ( (layer == 0) ? 0 : weight_count_prefix[layer-1]) + right_pos * layer_sizes[layer] + left_pos; }
	inline int get_layer_count() const { return layer_sizes.size(); }
	inlien int get_bias_index(int layer, int pos) const { return pos + neuron_count_prefix[layer-1] - layer_sizes[0] }; };
};
