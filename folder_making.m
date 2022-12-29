ims = dir(['D:\Plant Disease Dataset\lara2018-master\classifier\dataset\leaf\','*.jpg']);
for i=1:length(ims)
    I = imread(['D:\Plant Disease Dataset\lara2018-master\classifier\dataset\leaf\',ims(i).name]);
    id = str2num(ims(i).name(1:end-4));
    severity = datasetfull{id,7};
    if severity==0
        imwrite(I,['D:\Plant Disease Dataset\severity\0\',ims(i).name(1:end-4),'.jpg']);
    elseif severity==1
        imwrite(I,['D:\Plant Disease Dataset\severity\1\',ims(i).name(1:end-4),'.jpg']);
    elseif severity==2
        imwrite(I,['D:\Plant Disease Dataset\severity\2\',ims(i).name(1:end-4),'.jpg']);
    elseif severity==3
        imwrite(I,['D:\Plant Disease Dataset\severity\3\',ims(i).name(1:end-4),'.jpg']);
    elseif severity==4
        imwrite(I,['D:\Plant Disease Dataset\severity\4\',ims(i).name(1:end-4),'.jpg']);
    end
    i
end