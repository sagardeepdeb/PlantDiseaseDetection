gt = [ones(1,239) 2*ones(1,113) 3*ones(1,48)];

data = [data1(2,:) data2(2,:) data2(2,:)];

label = [];

for i=1:400;
    if data(i)<0.0457
        label = [label 1];
    elseif data(i)>0.0457 && data(i)<0.0827
        label = [label 2];
    else
        label = [label 3];
    end
end

accuracy = sum(label==gt)/size(label,2);