#include <stdio.h>
#include "mex.h"
#include "matrix.h"

#define DATA_TYPE double

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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/***********************INPUT DATA*******************************************/
	//Weight matrices
	const mxArray *mWeights = prhs[0];
	//Bias matrices
	const mxArray *mBiases = prhs[1];
	//Activation functions
	const mxArray *mActivation = prhs[2];
	//Name of file with training data
	char *trainingFileName;
	//Maximal number of epochs
	unsigned int epochs;
	//Desired error
	DATA_TYPE error;
	//Algorithm type
	unsigned int algorithm;
	//Report interval
	unsigned int report_interval;
	//Pointers to weights
	DATA_TYPE **weights;
	DATA_TYPE **biases;
	//Pointers to weight arrays
	mxArray **mAWeights;
	mxArray **mABiases;
	//Dimensions of weight arrays
	unsigned int *weightMDim;
	unsigned int *weightNDim;
	unsigned int *biasDim;
	//Number of layers
	unsigned int layers;
	//Activation type
	unsigned int *activation;
	//Output
	unsigned int *times;//Time passed
	DATA_TYPE *MSEs;//Mean squared errors
	unsigned int *entries;//Number of entries
	unsigned int mLogSize[2];
	//Iterators
	int i;
	//Output data
	mxArray *mMSEs;
	mxArray *mTimes;
	//Maximal execution time
	unsigned int max_time;
	//Passing parameters
	train_params params;
	/*******************************BODY*****************************************/
	nlhs = 0;
	if(nrhs != 9){
		mexErrMsgTxt("Must be 8 input arguments!");
		return;
	}
	trainingFileName = (char *)mxCalloc(mxGetN(prhs[3]) + 2, sizeof(char));
	mxGetString(prhs[3], trainingFileName, mxGetN(prhs[3]) + 2);
	epochs = (unsigned int) mxGetScalar(prhs[6]);
	error = (DATA_TYPE) mxGetScalar(prhs[5]);
	algorithm = (unsigned int) mxGetScalar(prhs[4]);
	report_interval = (unsigned int) mxGetScalar(prhs[7]);
	max_time = (unsigned int) mxGetScalar(prhs[8]);

	layers = mxGetNumberOfFields(mWeights);

	weightNDim = (unsigned int*)mxCalloc(layers, sizeof(unsigned int));
	weightMDim = (unsigned int*)mxCalloc(layers, sizeof(unsigned int));
	biasDim = (unsigned int*)mxCalloc(layers, sizeof(unsigned int));
	weights = (DATA_TYPE **)mxCalloc(layers, sizeof(DATA_TYPE *));
	biases = (DATA_TYPE **)mxCalloc(layers, sizeof(DATA_TYPE *));
	mAWeights = (mxArray **)mxCalloc(layers, sizeof(mxArray *));
	mABiases = (mxArray **)mxCalloc(layers, sizeof(mxArray *));
	
	activation = (unsigned int *)mxGetData(mActivation);

	for(i = 0; i < layers; i++){
		mAWeights[i] = (mxArray *)mxGetFieldByNumber(mWeights, 0, i);
		mABiases[i] = (mxArray *)mxGetFieldByNumber(mBiases, 0, i);
		weights[i] = (DATA_TYPE *)mxGetData(mAWeights[i]);
		biases[i] = (DATA_TYPE *)mxGetData(mABiases[i]);
		weightMDim[i] = mxGetN(mAWeights[i]);//Convert dimensions column-line storage
		weightNDim[i] = mxGetM(mAWeights[i]);
		biasDim[i] = mxGetM(mABiases[i]);
	}
	MSEs = (DATA_TYPE *)mxCalloc(epochs, sizeof(DATA_TYPE));
	times = (unsigned int *)mxCalloc(epochs, sizeof(unsigned int));
	entries = (unsigned int *)mxCalloc(1, sizeof(unsigned int));
	
	params.MSEs = MSEs;
	params.times = times;
	params.entries = entries;
	params.weights = weights;
	params.biases = biases;
	params.layers = layers;
	params.WMs = weightMDim;
	params.WNs = weightNDim;
	params.BNs = biasDim;
	params.activation = activation;
	params.trainingFileName = trainingFileName;
	params.algorithm = algorithm;
	params.error = error;
	params.epochs = epochs;
	params.report_interval = report_interval;
	params.max_time = max_time;

	fann_train_matlab(&params);
	/***********************OUTPUT DATA******************************************/
	mLogSize[0] = *entries;
	mLogSize[1] = 1;
	
	mMSEs = mxCreateNumericArray(2, mLogSize, mxSINGLE_CLASS, mxREAL);
	mTimes = mxCreateNumericArray(2, mLogSize, mxINT32_CLASS, mxREAL);
	mxSetData(mMSEs, MSEs);
	mxSetData(mTimes, times);

	nlhs = 2;
	plhs[0] = mMSEs;
	plhs[1] = mTimes;
	mxFree(trainingFileName);
	mxFree(weightMDim);
	mxFree(weightNDim);
	mxFree(biasDim);
	mxFree(weights);
	mxFree(biases);
}