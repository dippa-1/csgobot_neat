#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	int num_layers;
	int* topology;
	float*** weights;
	float** layers;
} NeuralNet;

void random_vector(float* vector, int size);

NeuralNet create_neural_net(int num_layers, int* topogoly);
void delete_neural_net(NeuralNet* net); 
void predict(NeuralNet* net, float* input, float* output);
void mutate_weights(NeuralNet* net, float mutation_rate);
NeuralNet* copy_weights(NeuralNet* dest, NeuralNet* src);
NeuralNet copy_neural_net(NeuralNet* net);

int main(void) {
	srand(time(NULL));

	float input[] = {1.0f, 1.0f, 1.0f};
	int topology[] = {3, 2, 2};
	NeuralNet net = create_neural_net(3, topology);

	float output[2];
	predict(&net, input, output);
	printf("result: %10f%10f\n", output[0], output[1]);

	mutate_weights(&net, 0.5f);
	predict(&net, input, output);
	printf("result: %10f%10f\n", output[0], output[1]);
	
	NeuralNet net2 = copy_neural_net(&net);
	predict(&net2, input, output);
	printf("result: %10f%10f\n", output[0], output[1]);

	delete_neural_net(&net);

	return 0;
}

void random_vector(float* vector, int size) {
	printf("Random vector: ");
	for (int i = 0; i < size; ++i) {
		vector[i] = (float)( rand() )/(float)( RAND_MAX ) * 2.0f - 1.0f;
		printf("%10f", vector[i]);
	}
	printf("\n");
}


NeuralNet create_neural_net(int num_layers, int* topology) {
	NeuralNet net = {num_layers, topology};
	// Every layer except for the last has weights
	net.weights = (float***) malloc(sizeof(float*) * num_layers-1);
	net.layers = (float**) malloc(sizeof(float*) * num_layers);
	for (int i = 0; i < num_layers; ++i) {
		// Create neurons for this layer
		float* l = (float*) calloc(topology[i], sizeof(float));
		net.layers[i] = l;
		
		if (i == num_layers - 1) break;

		// Create weights for this layer. Last layer cannot have weights
		float** weight_matrix = (float**) malloc(sizeof(float*) * topology[i]);
		net.weights[i] = weight_matrix;
		for (int j = 0; j < topology[i]; ++j) {
			float* weights_for_one_neuron = (float*) malloc(sizeof(float) * topology[i+1]);
			random_vector(weights_for_one_neuron, topology[i+1]);
			net.weights[i][j] = weights_for_one_neuron;
		}
		printf("\n");
	}

	return net;
}

void delete_neural_net(NeuralNet* net) {
	for (int i = 0; i < net->num_layers; ++i) {
		free(net->layers[i]);
		if (i == net->num_layers - 1) break;
		for (int j = 0; j < net->topology[i]; ++j) {
			free(net->weights[i][j]);
		}
		free(net->weights[i]);
	}
	free(net->layers);
	free(net->weights);
}


void predict(NeuralNet* net, float* input, float* output) {
	// copy input to first layer
	memcpy(net->layers[0], input, sizeof(float) * net->topology[0]);
	for (int i = 0; i < net->num_layers - 1; ++i) {
		// zero next layer
		memset(net->layers[i+1], 0, sizeof(float) * net->topology[i+1]);
		for (int j = 0; j < net->topology[i]; ++j) {
			for (int k = 0; k < net->topology[i+1]; ++k) {
				net->layers[i+1][k] += net->layers[i][j] * net->weights[i][j][k];
			}
		}
	}
	memcpy(output, net->layers[net->num_layers-1], sizeof(float) * net->topology[net->num_layers-1]);
}


void mutate_weights(NeuralNet* net, float mutation_rate) {
	for (int i = 0; i < net->num_layers-1; ++i) {
		for (int j = 0; j < net->topology[i]; ++j) {
			for (int k = 0; k < net->topology[i+1]; ++k) {
				if ( (rand() % 100) / 100.0f < mutation_rate) {
					net->weights[i][j][k] += ( rand() % 20 - 10) / 10.0f;
				}
			}
		}
	}
}


NeuralNet* copy_weights(NeuralNet* dest, NeuralNet* src) {
	// check if nets have the same topology
	if (dest->num_layers != src->num_layers) return NULL;
	for (int i = 0; i < dest->num_layers; ++i) {
		if (dest->topology[i] != src->topology[i]) return NULL;
	}

	// copy weights
	for (int i = 0; i < dest->num_layers-1; ++i) {
		for (int j = 0; j < dest->topology[i]; ++j) {
			memcpy(dest->weights[i][j], src->weights[i][j], sizeof(float) * dest->topology[i+1]);
		}
	}
	return dest;
}


NeuralNet copy_neural_net(NeuralNet* net) {
	NeuralNet new_net = {.num_layers = net->num_layers};
	
	new_net.topology = (int*) malloc(sizeof(int) * net->num_layers);
	memcpy(new_net.topology, net->topology, sizeof(int) * net->num_layers);
	
	new_net.weights = (float***) malloc(sizeof(float*) * net->num_layers - 1);
	new_net.layers = (float**) malloc(sizeof(float*) * net->num_layers);
	for (int i = 0; i < net->num_layers; ++i) {
		// Create neurons for this layer
		float* l = (float*) calloc(net->topology[i], sizeof(float));
		new_net.layers[i] = l;
		
		if (i == new_net.num_layers - 1) break;

		// Create weights for this layer. Last layer cannot have weights
		float** weight_matrix = (float**) malloc(sizeof(float*) * new_net.topology[i]);
		new_net.weights[i] = weight_matrix;
		for (int j = 0; j < new_net.topology[i]; ++j) {
			float* weights_for_one_neuron = (float*) malloc(sizeof(float) * new_net.topology[i+1]);
			new_net.weights[i][j] = weights_for_one_neuron;
		}
	}
	if (copy_weights(&new_net, net) == NULL) {
		printf("Failed to copy net: failed to copy weights\n");
		exit(1);
	}

	return new_net;
}
