#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void random_vector(float* vector, int size);

void predict(float* input, float* weights, int size, float* output) {
	output[0] = 0.0f;
	for (int i = 0; i < size; ++i) {
		output[0] += input[i] * weights[i];
	}
	printf("Prediction: %f\n", output[0]);
}

int main(void) {
	srand(time(NULL));

	float input[4];
	float weights[4];
	random_vector(input, 4);
	random_vector(weights, 4);

	float output[1];
	predict(input, weights, 4, output);

	return 0;
}

void random_vector(float* vector, int size) {
	printf("Random vector: ");
	for (int i = 0; i < size; ++i) {
		vector[i] = (float)( rand() )/(float)( RAND_MAX ) * 2.0f - 1.0f;
		printf("%f ", vector[i]);
	}
	printf("\n");
}
