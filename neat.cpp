#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#define WIDTH 800
#define HEIGHT 600

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

class Target {
	private:
	public:
	std::vector<float> pos;
	float radius;

	Target() {
		this->pos = { rand() % (WIDTH/2) + (WIDTH/2.0f), rand() % (HEIGHT/2) + (HEIGHT/2.0f)};
		this->radius = 20.0f;
	}
};

class Player {
	private:

	public:
	NeuralNet brain;
	float radius;
	std::vector<float> pos;
	std::vector<float> vel;
	float fitness;

	Player(std::vector<int> brain_topology): brain(brain_topology) {
		this->fitness = 0.0f;
		this->radius = 20.0f;
		this->pos = {50.0f, 50.0f};
		this->vel = {0.0f, 0.0f};
	}

	Player(const Player& other): brain(other.brain) {
		this->fitness = 0.0f;
		this->radius = 20.0f;
		this->pos = {50.0f, 50.0f};
		this->vel = {0.0f, 0.0f};
	}

	void update(void) {
		this->pos[0] += this->vel[0];
		this->pos[1] += this->vel[1];
		if (this->pos[1] < -500.0f) this->pos[1] = -500.0f;
		else if (this->pos[1] > 500.0f + HEIGHT) this->pos[1] = 500.0f + HEIGHT;
		this->vel[0] *= 0.75f;
		this->vel[1] *= 0.75f;
	}

	void mutate(float rate) {
		this->brain.mutate_weights(rate);
	}

	void think(Target& target) {
		auto prediction = this->brain.predict({
			(target.pos[0]-this->pos[0])/WIDTH, 
			(target.pos[1]-this->pos[1])/HEIGHT, 
			(this->vel[0]-target.pos[0]+this->pos[0])/WIDTH, 
			(this->vel[1]-target.pos[1]+this->pos[1])/HEIGHT
			});
		this->vel[0] += prediction[0]*WIDTH/300;
    	this->vel[1] += prediction[1]*HEIGHT/300;
	}
};

class Population {
	private:

	public:
	int generation;
	float average_fitness;
	float best_fitness;
	std::vector<Player> players;
	Population(int size, std::vector<int> topology) {
		for (int i = 0; i < size; ++i) {
			players.push_back(Player(topology));
		}
		generation = 1;
	}

	Population(const Population& pop): generation(generation), average_fitness(average_fitness), best_fitness(best_fitness) {
		for (auto player : pop.players) {
			players.push_back(Player(player));
		}
	}

	void next_generation(float mutation_rate) {
		//printf("Generation %d\n", generation);
		float fitness_sum = 0.0f;
		this->best_fitness = 0.0f;
		for (int i = 0; i < this->players.size(); ++i) {
			fitness_sum += players[i].fitness;
			if (this->players[i].fitness > this->best_fitness) this->best_fitness = this->players[i].fitness;
			//printf("Fitness %f\n", players[i].fitness);
		}
		this->average_fitness = fitness_sum / this->players.size();
		//printf("Avg Fitness %f\n", average_fitness);
		for (int i = 0; i < this->players.size(); ++i) {
			this->players[i].fitness /= fitness_sum;
		}
		//printf("Best Fitness %f\n", best_fitness);
		std::vector<float> chances_lut(this->players.size());
		chances_lut[0] = this->players[0].fitness;
		for (int i = 1; i < this->players.size(); ++i) {
			chances_lut[i] = chances_lut[i-1] + this->players[i].fitness;
		}
		// Save old players
		Population old_pop(*this);
		for (int i = 0; i < this->players.size(); ++i) {
			float random = (float) rand() / RAND_MAX;
			int index = 0;
			while (random > chances_lut[index]) ++index;
			this->players[i] = Player(old_pop.players[index]);
			this->players[i].mutate(mutation_rate);
		}
		++this->generation;
	}
};



// NEAT functions
float calculate_reward(std::vector<float> output, std::vector<float> goal);

void sketch_static(void) {
	FILE* logfile = fopen("log.txt", "w");
	fprintf(logfile, "#Generation\tBest\tAverage\n");

	int frameCount = 1;

	// New Population
	Population pop(20, {4,4,2});
	Target target;

	// Simulate Generations
	while (pop.generation <= 100) {

		// Update each player
		for (int i = 0; i < pop.players.size(); ++i) {
			// Update velocity with neural network
			pop.players[i].think(target);
			// move player in direction
			pop.players[i].update();
			// Calculate Distance to target
			const float dist = sqrtf((pop.players[i].pos[0] - target.pos[0])*(pop.players[i].pos[0] - target.pos[0])
				+ (pop.players[i].pos[1] - target.pos[1])*(pop.players[i].pos[1] - target.pos[1])
				);
			// Calculate Fitness
			float fitness = target.radius / dist / 2.0f;
			if (fitness > 1.0f) fitness = 1.0f;
			if (dist < target.radius) fitness = 1.0f;
			pop.players[i].fitness += fitness;
			//printf("calculated %f fitness. Player now has %f\n", fitness, pop.players[i].fitness);
		}

		// the rate of which the target changes its position increases over time
		int tmp = (int)(200.0 / (1 + exp(-1 + 0.03*pop.generation)));
		if (tmp < 5) tmp = 5;
		if (frameCount % tmp == 0 && frameCount % 150 != 0) target = Target();

		if (frameCount % 150 == 0) {
			// next generation
			pop.next_generation(0.1);
			fprintf(logfile, "%d\t%f\t%f\n", pop.generation-1, pop.best_fitness, pop.average_fitness);
			// new target
			target = Target();
			float new_startpos[2] = {(float) (rand() % WIDTH), (float) (rand() % HEIGHT)};
			for (int i = 0; i < pop.players.size(); ++i) {
				pop.players[i].pos = {new_startpos[0], new_startpos[1]};
			}
		}

		++frameCount;

	}

	fclose(logfile);
}


int main(void) {
	srand(time(NULL));

	sketch_static();

	/*std::vector<float> input(3);
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
			gen_fitness += reward;
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

	fclose(logfile);*/

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
