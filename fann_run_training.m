%load('alphabet.mat');
%SET_SIZE = 430000;
%ITERATIONS = 4;
%delete 'net.ssv';
%[alphabet, targets] = shuffle_trainset(alphabet, targets);
%net = newff(alphabet, targets, [120 120]);
%net = init(net);
%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'logsig';
%net.layers{3}.transferFcn = 'logsig';
%global_log = [0 0 0];
%
%RPROP: inc_factor, dec_factor, delta_min, delta_max, delta_zero
%QPROP: mu, decay
%
for i=1:ITERATIONS   
    net.userdata.algorithm = 'FANN_TRAIN_BATCH';
    net.userdata.report_interval = 1;
    net.trainParam.goal = 1e-8;
    net.trainParam.epochs = 190;
    net.trainParam.time = 6900;
 
    [net, log] = fann_train(net, alphabet(:,1:SET_SIZE), targets(:,1:SET_SIZE));
    global_log = connect_logs(global_log, log, 'fann');
    if min(log(:,2)) < net.trainParam.goal 
    	break;
    end

    net.trainFcn = 'trainscg';
    net.divideFcn = 'dividetrain';
    net.trainParam.goal = 1e-8;
    net.trainParam.epochs = 190;
    net.trainParam.time = 6900;
    net.trainParam.max_fail = 1000;
    net.trainParam.min_grad = 1e-10;
    
    [net, log] = train(net, alphabet(:,1:SET_SIZE), targets(:,1:SET_SIZE));
    global_log = connect_logs(global_log, log, 'matlab');
    if min(log.perf(:,2)') < net.trainParam.goal 
    	break;
    end
end
    %global_log(1,:)=[];

