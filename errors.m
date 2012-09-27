ST = 430001;
EN = 433550;
res = sim(net, alphabet(:,ST:EN));
[v1 i1] = max(res);
[v2 i2] = max(targets(:,ST:EN));
Error = 0;
for i=1:length(i1)
    if i1(i) ~= i2(i)
        Error = Error +1;
    end
end