function [alphabet, targets] = shuffle_trainset(alphabet, targets)
%Shuffle trainset
TRAIN_SIZE = size(targets, 2);
for i=1:TRAIN_SIZE
    src = randi(TRAIN_SIZE);
    dst = randi(TRAIN_SIZE);
    tmp = alphabet(:, dst);
    alphabet(:, dst) = alphabet(:, src);
    alphabet(:, src) = tmp;
    tmp = targets(:, dst);
    targets(:, dst) = targets(:, src);
    targets(:, src) = tmp;
end
