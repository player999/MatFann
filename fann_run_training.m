%load('alphabet.mat');
SET_SIZE = 430000;
ITERATIONS = 1;

net = newff(alphabet, targets, [130 100]);
net = init(net);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'logsig';

for i=1:ITERATIONS   
    net.userdata.algorithm = 'FANN_TRAIN_BATCH';
    net.userdata.report_interval = 10;
    net.trainParam.goal = 1e-8;
    net.trainParam.epochs = 100000;
    net.trainParam.time = 27000;

    [net, log1] = fann_train(net, alphabet(:,1:SET_SIZE), targets(:,1:SET_SIZE));
    
    net.trainFcn = 'trainscg';
    net.trainParam.max_fail = 500000;
    net.trainParam.min_grad = 1e-20;
    net.trainParam.goal = 1e-8;
    net.trainParam.epochs = 100000; 
    net.trainParam.time = 27000;
    [net, log2] = train(net, alphabet(:,1:SET_SIZE), targets(:,1:SET_SIZE));
    
end


save('training_output.mat', 'net', 'log1', 'log2');
