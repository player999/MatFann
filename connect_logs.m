function log = connect_logs(log1, log2, type)
    if strcmp(type, 'fann')
        log2(:,1) = log2(:,1) + max(log1(:,1));
        log2(:,3) = log2(:,3) + max(log1(:,3));
        log = cat(1, log1, log2);
    elseif strcmp(type, 'matlab')
        log3 = cat(2, log2.epoch', log2.perf', log2.time');
        log3(:,1) = log3(:,1) + max(log1(:,1));
        log3(:,3) = log3(:,3) + max(log1(:,3));
        log = cat(1, log1, log3);
    end
end