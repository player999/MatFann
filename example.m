load('alphabet.mat');
net = newff(alphabet, targets, [2 2]);
net = init(net);

net.userdata.algorithm = 'FANN_TRAIN_RPROP';
net.trainParam.goal = 1e-20;
net.trainParam.epochs = 10000000;
net.userdata.report_interval = 10000;
net.userdata.time = 1;
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'logsig';
[alphabet, targets] = shuffle_trainset(alphabet, targets);
%save('shuffled_alphabet.mat', 'alphabet', 'targets');
%net.trainFcn = 'trainscg';
[net, log] = fann_train(net, alphabet, targets);
%[net, log2] = train(net, alphabet, targets);
%save('training_output.mat', 'net', 'log');
