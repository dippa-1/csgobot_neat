#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <map>

class NeuralNet {
    private:
    std::vector<int> topology;
    std::vector< std::vector <std::vector<float>>> weights;
	std::vector< std::vector<float>> layers;

    public:
    NeuralNet(std::vector<int> topology): topology(topology) {
		for (int i = 0; i < topology.size()-1; ++i) {
			std::vector< std::vector<float>> weightlayer;
			for (int j = 0; j < topology[i]; ++j) {
				std::vector<float> w;
				for (int k = 0; k < topology[i+1]; ++k) {
					w.push_back( (float)( rand() )/(float)( RAND_MAX ) * 4.0f - 2.0f );
				}
				weightlayer.push_back(w);
			}
			weights.push_back(weightlayer);
			std::vector<float> layer(topology[i]);
			layers.push_back(layer);
		}
		// final layer (output)
		layers.push_back(std::vector<float>(*(topology.end())));
    }

	std::vector<float> predict(std::vector<float> input) {
		for (int i = 0; i < this->layers.size() - 1; ++i) {
			// zero next layer
			std::fill(this->layers[i].begin(), this->layers[i].end(), 0.0f);
			for (int j = 0; j < this->topology[i]; ++j) {
				for (int k = 0; k < this->topology[i+1]; ++k) {
					this->layers[i+1][k] += this->layers[i][j] * this->weights[i][j][k];
				}
			}
		}
		return std::vector(this->layers[*topology.end()]);
	}

	void mutate_weights(float rate) {
		for (int i = 0; i < this->layers.size()-1; ++i) {
			for (int j = 0; j < this->topology[i]; ++j) {
				for (int k = 0; k < this->topology[i+1]; ++k) {
					if ( (rand() % 10000) / 10000.0f < rate) {
						this->weights[i][j][k] += 2.0f * ( rand() - RAND_MAX/2.0f) / RAND_MAX;
					}
				}
			}
		}
	}
};

class Population {
	private:
	std::vector<NeuralNet> nets;
	float average_fitness;
	float best_fitness;
	std::vector<float> fitness;

	public:
	Population(int size, std::vector<int> topology) {
		for (int i = 0; i < size; ++i) {
			nets.push_back(NeuralNet(topology));
		}
	}

	void next_generation(void) {
		Population old_pop(*this);


	}
};


// NEAT functions
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
		printf("best fitness = %.4f from %d\n", maxreward, best_person);
		fprintf(logfile, "%d\t%f\n", gen, maxreward);
		float chances[pop.size];
		for (int i = 0; i < pop.size; ++i) {
			chances[i] = rewards[i] / gen_fitness;
		}
		if (gen < gensize-1) {
			pop = inherit_population(&pop, chances);
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
