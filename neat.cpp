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
		layers.push_back(std::vector<float>(topology.back()));
    }

	std::vector<float> predict(std::vector<float> input) {
		this->layers[0] = std::vector<float>(input);
		for (int i = 0; i < this->layers.size() - 1; ++i) {
			// zero next layer
			std::fill(this->layers[i+1].begin(), this->layers[i+1].end(), 0.0f);
			for (int j = 0; j < this->topology[i]; ++j) {
				for (int k = 0; k < this->topology[i+1]; ++k) {
					this->layers[i+1][k] += this->layers[i][j] * this->weights[i][j][k];
				}
			}
		}
		return std::vector<float>(this->layers[this->layers.size()-1]);
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
	float average_fitness;
	float best_fitness;

	public:
	std::vector<float> fitness;
	std::vector<NeuralNet> nets;
	Population(int size, std::vector<int> topology) {
		for (int i = 0; i < size; ++i) {
			nets.push_back(NeuralNet(topology));
		}
		this->fitness = std::vector<float>(size);
	}

	void next_generation(float mutation_rate) {
		Population old_pop(*this);
		float fitness_sum = 0.0f;
		for (float f : this->fitness) {
			fitness_sum += f;
		}
		for (int i = 0; i < this->fitness.size(); ++i) {
			this->fitness[i] /= fitness_sum;
		}
		std::vector<float> chances_lut(this->nets.size());
		chances_lut[0] = this->fitness[0];
		for (int i = 1; i < this->fitness.size(); ++i) {
			chances_lut[i] = chances_lut[i-1] + this->fitness[i];
		}
		for (int i = 0; i < this->fitness.size(); ++i) {
			float random = (float) rand() / RAND_MAX;
			int index = 0;
			while (random > chances_lut[index]) ++index;
			this->nets[i] = NeuralNet(old_pop.nets[index]);
			this->nets[i].mutate_weights(mutation_rate);
		}
	}
};


// NEAT functions
float calculate_reward(std::vector<float> output, std::vector<float> goal);


int main(void) {
	srand(time(NULL));

	std::vector<float> input(3);
	std::vector<float> input2(3);
	std::vector<float> input3(3);

	FILE* logfile = fopen("log.txt", "w");
	fprintf(logfile, "#Generation\tBest fitness\n");

	int popsize = 50;
	int gensize = 100;
	Population pop = Population(popsize, {3,2});
	for (int gen = 0; gen < gensize; ++gen) {
		float maxreward = 0.0f;
		int best_person = 0;
		for (int i = 0; i < input.size(); ++i) {
			input[i] = (float)rand() / RAND_MAX * 1000.0f - 500.0f;
			input2[i] = (float)rand() / RAND_MAX * 1000.0f - 500.0f;
			input3[i] = (float)rand() / RAND_MAX * 1000.0f - 500.0f;
		}
		std::vector<float> goal = {input[0] + input[1] + input[2], 5*input[2] - input[1] - input[0]};
		std::vector<float> goal2 = {input2[0] + input2[1] + input2[2], 5*input2[2] - input2[1] - input2[0]};
		std::vector<float> goal3 = {input3[0] + input3[1] + input3[2], 5*input3[2] - input3[1] - input3[0]};
		float gen_fitness = 0.0f;
		for (int i = 0; i < popsize; ++i) {
			std::vector<float> output = pop.nets[i].predict(input);
			std::vector<float> output2 = pop.nets[i].predict(input2);
			std::vector<float> output3 = pop.nets[i].predict(input3);
			float reward = calculate_reward(output, goal);
			reward += calculate_reward(output2, goal2);
			reward += calculate_reward(output3, goal3);
			pop.fitness[i] = reward;
			gen_fitness += reward;
			if (reward > maxreward) {
				maxreward = reward;
				best_person = i;
			}
		}
		fprintf(logfile, "%d\t%f\n", gen, maxreward);
		if (gen < gensize-1) {
			pop.next_generation(0.167f);
		} else {
			printf("Testing input 1 1 1\n");
			input = {1,1,1};
			goal = {input[0] + input[1] + input[2], 5*input[2] - input[1] - input[0]};
			printf("Goal: %f %f\n", input[0] + input[1] + input[2], 5*input[2] - input[1] - input[0]);
			std::vector<float> output = pop.nets[best_person].predict(input);
			printf("Output: %f %f\n", output[0], output[1]);
		}
	}

	fclose(logfile);

	return 0;
}


float calculate_reward(std::vector<float> output, std::vector<float> goal) {
	float reward = 1.0f;
	for (int i = 0; i < output.size(); ++i) {
		float r = 1.0f/fabs(goal[i]/output[i] - 1.0f);
		if (r > 1000.0f) r = 1000.0f;
		reward *= r;
	}
	reward = pow(reward, 1.0f/output.size());
	return reward;
}