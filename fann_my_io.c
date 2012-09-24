#include <stdlib.h>
#include <stdio.h>
#include <fann.h>
#include <mex.h>
#include "fann_my_io.h"

//#define DEBUG 1

void fann_save_matrices(struct fann *network, char *fname){
	unsigned int layers;
	unsigned int layer[100];
	unsigned int bias[100];
	unsigned int total_weights;
	unsigned int neuron_inputs;
	unsigned int writes_counter;
	float weight;
	struct 	fann_connection *connections;
	FILE *array;
	char array_name[255];
	int i, j, k;
	writes_counter = 0;
	layers = fann_get_num_layers(network);
	fann_get_layer_array(network, layer);
	fann_get_bias_array(network, bias);
	total_weights = fann_get_total_connections(network);
	printf("Total weights: %i\n", total_weights);
	connections = (struct 	fann_connection *)
				malloc(total_weights * sizeof(struct 	fann_connection));
	fann_get_connection_array(network, connections);
	for(i = 1; i < layers; i++){
		sprintf(array_name, "%s_W%i.net", fname, i);
		array = fopen(array_name, "wb");
		for (j = 0; j < layer[i]*(layer[i-1] + 1); j++){
			weight = connections[writes_counter].weight;
#ifdef DEBUG
			mexPrintf("Number:\t%i\n", writes_counter);
			mexPrintf("Weight:\t%e\n", connections[writes_counter].weight);
			mexPrintf("From:\t%i\n", connections[writes_counter].from_neuron);
			mexPrintf("To:\t%i\n", connections[writes_counter].to_neuron);
			mexEvalString("drawnow;");
#endif
			fwrite(&weight, sizeof(float) , 1, array);
			writes_counter++;
		}
		fclose(array);
	}
	return;
}

