function [net, log] = fann_train(net, alphabet, targets)
numLayers = net.numLayers;
%Extract weights (for future)
weights.l1 = single(cat(2, net.IW{1}, net.b{1}))';
biases.b1 = single(net.b{1});
for i = 2:numLayers
    weights.(['l' int2str(i)]) = single(cat(2, net.LW{i, i - 1}, net.b{i}))';
    biases.(['b' int2str(i)]) = single(net.b{i});
end
%Extract activation function
activation = zeros(1, numLayers);
for i = 1:numLayers
    tf = net.layers{i}.transferFcn;
    switch tf
        case 'logsig'
            activation(i) = 0;
        case 'tansig'
            activation(i) = 1;
        case 'purelin'
            activation(i) = 3;
        otherwise
            activation(i) = 0;
            warning('Undefined transfer function. Demoting to logsig.');
    end
end
%Prepare training set
fid = fopen('net.ssv','w');
fprintf(fid, '%i %i %i\n', size(alphabet,2), size(alphabet,1), size(targets,1));
for i=1:size(alphabet ,2)
    fprintf(fid, '%e ', alphabet(1:size(alphabet,1)-1,i));
    fprintf(fid, '%e\n', alphabet(size(alphabet,1),i));
    fprintf(fid, '%e ', targets(1:size(targets,1)-1,i));
    fprintf(fid, '%e\n', targets(size(targets,1),i));
end
fclose(fid);

%Set algorithm
switch net.userdata.algorithm
    case 'FANN_TRAIN_BATCH'
        algorithm = 0;
    case 'FANN_TRAIN_QUICKPROP'
        algorithm = 1;
    case 'FANN_TRAIN_RPROP'
        algorithm = 2;
    case 'FANN_TRAIN_INCREMENTAL'
        algorithm = 3;
    otherwise
        algorithm = 0;
        warning('Unknown algorithm. Demoting to FANN_TRAIN_BATCH');
end
desired_error = net.trainParam.goal;
epochs = net.trainParam.epochs;
report_interval = net.userdata.report_interval;
max_time = net.userdata.time;

[log1 log2] = fann_train_call(weights, biases, uint32(activation), 'net', uint32(algorithm), single(desired_error), uint32(epochs), uint32(report_interval), uint32(max_time));

log = zeros(length(log1), 3);
log(:,1) = [0:report_interval:(length(log1) - 1) * report_interval];
log(1,1) = 1;
log(:,2) = log1;
log(:,3) = log2;

for i=1:numLayers
    fid = fopen(['net_W' int2str(i) '.net'], 'r');
    if i == 1
        tmp = fread(fid, (net.layers{i}.dimensions + 1) * net.layers{i}.dimensions, 'float32');
        tmp = reshape(tmp, net.layers{i}.dimensions + 1, net.layers{i}.dimensions)';
        net.b{i} = tmp(:, net.layers{i}.dimensions + 1);
        net.IW{1} = tmp(:, 1:net.layers{i}.dimensions);
    else
        tmp = fread(fid, (net.layers{i-1}.dimensions + 1) * net.layers{i}.dimensions, 'float32');
        tmp = reshape(tmp, net.layers{i-1}.dimensions + 1, net.layers{i}.dimensions)';
        net.b{i} = tmp(:, net.layers{i-1}.dimensions + 1);
        net.LW{i, i-1} = tmp(:, 1:net.layers{i-1}.dimensions);
    end
    fclose(fid);
    net.inputs{1}.processFcns = {'removeconstantrows'};
    net.outputs{numLayers}.processFcns = {'removeconstantrows'};

end

