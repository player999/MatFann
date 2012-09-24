#include <stdlib.h>
#include <stdio.h>
#include <Windows.h>
#include <doublefann.h>
#include <mex.h>
#include "fann_my_io.h"

//#define DEBUG 1
#define DATA_TYPE double

typedef struct {
	time_t seconds;
	FILE *logFile;
	DATA_TYPE *MSE_log;
	unsigned int *time_log;
	unsigned int log_rows;
	unsigned int max_time;
	unsigned int start_time;
	char *training_fn; 
	char *outputNetFN;
}callback_params;

typedef struct {
	DATA_TYPE *MSEs;
	unsigned int *times;
	unsigned int *entries;
	DATA_TYPE **weights;
	DATA_TYPE **biases;
	unsigned int layers;
	unsigned int *WMs;
	unsigned int *WNs;
	unsigned int *BNs;
	unsigned int *activation;
	char * trainingFileName;
	unsigned int algorithm;
	DATA_TYPE error;
	unsigned int epochs;
	unsigned int report_interval;
	unsigned int max_time;
} train_params;


int FANN_API train_callback(struct fann *ann, struct fann_train_data *train,
                           unsigned int max_epochs, unsigned int epochs_between_reports,
                           DATA_TYPE desired_error, unsigned int epochs);

void init_weights(struct fann *network, train_params *params);

void fann_train_matlab(train_params *params)
{
	struct fann *network;
	struct fann_train_data *train_data;
	callback_params cb_params;

	unsigned int *layer_sizes;
	unsigned int fann_activation;
	unsigned int fann_algorithm;
	unsigned int i;

	char trainingDataFileName[255];
	char outputNetworkFileName[255];
	char logFileName[255];

	FILE *trainingDataFile;

	//Generate file names
	sprintf(trainingDataFileName, "%s.ssv", params->trainingFileName);
	sprintf(outputNetworkFileName, "%s.net", params->trainingFileName);
	sprintf(logFileName, "%s.log", params->trainingFileName);
	//Open log
	cb_params.logFile = fopen(logFileName, "w");
	
	//Get layer sizes from input
	layer_sizes = (unsigned int *)malloc((params->layers + 1) 
					* sizeof(unsigned int));
	layer_sizes[0] = params->BNs[0];
	for(i = 1; i < params->layers + 1; i++){
		layer_sizes[i] = params->BNs[i - 1];
	}
	network = fann_create_standard_array(params->layers + 1, layer_sizes);
	fann_set_error_log(network, cb_params.logFile);
	for(i = 1; i < params->layers + 1; i++){
		switch(params->activation[i - 1]){
		case 0:
			fann_activation = FANN_SIGMOID;
			break;
		case 1:
			fann_activation = FANN_SIGMOID_SYMMETRIC;
			break;
		case 2:
			fann_activation = FANN_SIGMOID_STEPWISE;
			break;
		case 3:
			fann_activation = FANN_LINEAR;
			break;
		default:
			fann_activation = FANN_SIGMOID;
			break;
		}
		fann_set_activation_function_layer(network, fann_activation, i);
	}
	
	fann_set_callback(network, train_callback);
	//Global variables for callback
	cb_params.outputNetFN = outputNetworkFileName;
	cb_params.MSE_log = params->MSEs;
	cb_params.time_log = params->times;
	cb_params.log_rows = 0;
	cb_params.training_fn = params->trainingFileName;
	cb_params.max_time = params->max_time;
	cb_params.start_time = time(NULL);

	switch(params->algorithm){
	case 0:
		fann_algorithm = FANN_TRAIN_BATCH;
		break;
	case 1:
		fann_algorithm = FANN_TRAIN_QUICKPROP;
		break;
	case 2:
		fann_algorithm = FANN_TRAIN_RPROP;
		break;
	case 3:
		fann_algorithm = FANN_TRAIN_INCREMENTAL;
		break;
	default:
		fann_algorithm = FANN_TRAIN_BATCH;
		break;
	}
	fann_set_training_algorithm(network, fann_algorithm);

	init_weights(network, params);

	train_data = fann_read_train_from_file(trainingDataFileName);
	cb_params.seconds = time(NULL);
	fann_set_user_data(network, (void *)&cb_params);
	fann_train_on_data(network, train_data, params->epochs, params->report_interval, params->error);
	
	fann_save(network, outputNetworkFileName);
	fann_save_matrices(network, params->trainingFileName);
	
	*(params->entries) = cb_params.log_rows;
	
	fann_destroy_train(train_data);
	fann_destroy(network);
	free(layer_sizes);
	fclose(cb_params.logFile);
	return;
}

int FANN_API train_callback(struct fann *ann, struct fann_train_data *train,
                           unsigned int max_epochs, unsigned int epochs_between_reports,
                           DATA_TYPE desired_error, unsigned int epochs)
{
	callback_params *cb_params;
	time_t secs;

	cb_params = (callback_params *)fann_get_user_data(ann);
	secs = time(NULL);
	cb_params->MSE_log[cb_params->log_rows] = fann_get_MSE(ann);
	cb_params->time_log[cb_params->log_rows++] = secs-cb_params->seconds;
	mexPrintf("%d, %15e, %ld\n", epochs, fann_get_MSE(ann), secs-cb_params->seconds);
	mexEvalString("drawnow;");
	fprintf(cb_params->logFile, "%d, %f, %ld\n", epochs, fann_get_MSE(ann), secs-cb_params->seconds);
	if(!(cb_params->log_rows % 10)){
		fann_save_matrices(ann, cb_params->training_fn);
		fann_save(ann, cb_params->outputNetFN);
	}
	if (utIsInterruptPending()){
		mexPrintf("Cought Ctrl+C. Wait a 5 sec.\n");
		mexEvalString("drawnow;");
		Sleep(5);
		utSetInterruptPending(0);
		return -1;
	}

	if ((secs - cb_params->start_time) > cb_params->max_time) return -1;

	return 0;
}

void init_weights(struct fann *network, train_params *params){
	unsigned int from, to;
	unsigned int i, j, k;
	unsigned int neuron_bias;
	unsigned int array_idx;

	neuron_bias = 0;
	for(i = 0; i < params->layers; i++){
		array_idx = 0;
		for(j = 0; j < params->WMs[i]; j++){
			for(k = 0; k < params->WNs[i]; k++){
				from = k + neuron_bias;
				to = j + neuron_bias + params->WNs[i];
#ifdef DEBUG
				mexPrintf("%i -> %i = %e, idx=%i\n", from, to, params->weights[i][array_idx], array_idx);
#endif
				fann_set_weight(network, from, to, params->weights[i][array_idx++]);
				mexEvalString("drawnow;");
			}
		}
		neuron_bias += params->WNs[i];
	}
}
int main(){
	return 0;
}
