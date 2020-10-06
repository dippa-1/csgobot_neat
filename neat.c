#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
	int num_layers;
	int* topology;
	float*** weights;
	float** layers;
} NeuralNet;

typedef struct {
	int size;
	NeuralNet** people;
} Population;

// Mathematical help functions
void random_vector(float* vector, int size);

// Neural net functions
NeuralNet create_neural_net(int num_layers, int* topogoly);
void delete_neural_net(NeuralNet* net); 
void predict(NeuralNet* net, float* input, float* output);
void mutate_weights(NeuralNet* net, float mutation_rate);
NeuralNet* copy_weights(NeuralNet* dest, NeuralNet* src);
NeuralNet copy_neural_net(NeuralNet* net);

// NEAT functions
Population new_population(int size);
void delete_population(Population* population);
float calculate_reward(float* output, float* goal, int vector_size);
Population inherit_population(Population* precesters, float* chances_to_inherit);


int main(void) {
	srand(time(NULL));

	float input[5] = {};

	FILE* logfile = fopen("log.txt", "w");
	fprintf(logfile, "#Generation\tBest fitness\n");

	float output[2];
	int popsize = 50;
	int gensize = 100;
	Population pop = new_population(popsize);
	for (int gen = 0; gen < gensize; ++gen) {
		float maxreward = 0.0f;
		int best_person = 0;
		for (int i = 0; i < 5; ++i) {
			input[i] = (float)rand() / RAND_MAX * 1000.0f - 500.0f;
		}
		float goal[2] = {input[0] + 2*input[1] + 3*input[2] + 4*input[3] + 5*input[4], input[3] - input[2]};
		float gen_fitness = 0.0f;
		float rewards[pop.size];
		for (int i = 0; i < popsize; ++i) {
			predict(pop.people[i], input, output);
			float reward = calculate_reward(pop.people[i]->layers[1], goal, 2);
			gen_fitness += reward;
			rewards[i] = reward;
			if (reward > maxreward) {
				maxreward = reward;
				best_person = i;
			}
		}
		predict(pop.people[best_person], input, output);
		//printf("best result: %+7.3f\n", output[0]);
		//printf("best weights: %+5.3f%+7.3f\n", pop.people[best_person]->weights[0][0][0], pop.people[best_person]->weights[0][1][0]);
		printf("best fitness = %.4f from %d\n", maxreward, best_person);
		fprintf(logfile, "%d\t%f\n", gen, maxreward);
		float chances[pop.size];
		for (int i = 0; i < pop.size; ++i) {
			chances[i] = rewards[i] / gen_fitness;
		}
		if (gen < gensize-1) {
			pop = inherit_population(&pop, chances);
			// inherit and mutate
			/*for (int i = 0; i < popsize; ++i) {
				if (i == best_person) continue;
				delete_neural_net(pop.people[i]);
				*(pop.people[i]) = copy_neural_net(pop.people[best_person]);
				mutate_weights(pop.people[i], 0.5);
			}*/
		} else {
			printf("Testing input 5 4 3 2 1\n");
			goal[0] = 0.0f;
			goal[1] = 0.0f;
			input[0] = 5;
			input[1] = 4;
			input[2] = 3;
			input[3] = 2;
			input[4] = 1;
			goal[0] = input[0] + 2*input[1] + 3*input[2] + 4*input[3] + 5*input[4];
			goal[1] = input[3] - input[2];
			printf("Goal: %f %f\n", goal[0], goal[1]);
			predict(pop.people[best_person], input, output);
			printf("Output: %f %f\n", output[0], output[1]);
		}
	}

	fclose(logfile);
	delete_population(&pop);

	return 0;
}

void random_vector(float* vector, int size) {
	for (int i = 0; i < size; ++i) {
		vector[i] = (float)( rand() )/(float)( RAND_MAX ) * 2.0f - 1.0f;
	}
}


NeuralNet create_neural_net(int num_layers, int* topology) {
	NeuralNet net = {num_layers};
	net.topology = (int*) malloc (sizeof(int) * num_layers);
	memcpy(net.topology, topology, num_layers*sizeof(int));
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
	free(net->topology);
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
				if ( (rand() % 10000) / 10000.0f < mutation_rate) {
					net->weights[i][j][k] += 2.0f * ( rand() - RAND_MAX/2.0f) / RAND_MAX;
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


Population new_population(int size) {
	Population pop = {.size = size};
	int topology[] = {5, 2};
	pop.people = (NeuralNet**) malloc(sizeof(NeuralNet*) * size);
	for (int i = 0; i < size; ++i) {
		pop.people[i] = (NeuralNet*) malloc(sizeof(NeuralNet));
		*(pop.people[i]) = create_neural_net(2, topology);
	}
	return pop;
}

void delete_population(Population* population) {
	for (int i = 0; i < population->size; ++i) {
		delete_neural_net(population->people[i]);
		free(population->people[i]);
	}
	free(population->people);
}

float calculate_reward(float* output, float* goal, int vector_size) {
	// calculates reward like a sigmoid function
	float reward = 0.0f;
	for (int i = 0; i < vector_size; ++i) {
		float r = 1.0f/fabs(goal[i] - output[i]);
		if (r > 1.0f) r = 1.0f;
		reward += r;
	}
	return reward;
}

Population inherit_population(Population* precesters, float* chances_to_inherit) {
	Population new_pop = new_population(precesters->size);
	float chances_lut[precesters->size];
	chances_lut[0] = chances_to_inherit[0];
	for (int i = 1; i < precesters->size; ++i) {
		chances_lut[i] = chances_lut[i-1] + chances_to_inherit[i];
	}
	for (int i = 0; i < precesters->size; ++i) {
		float random = (float) rand() / RAND_MAX;
		int index = 0;
		while (random > chances_lut[index]) ++index;
		delete_neural_net(new_pop.people[i]);
		*(new_pop.people[i]) = copy_neural_net(precesters->people[index]);
		mutate_weights(new_pop.people[i], 0.1);
	}
	delete_population(precesters);
	return new_pop;
}
