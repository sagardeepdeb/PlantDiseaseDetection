ims = dir(['C:\Users\rachi\Downloads\Seg_Dataset\coffee-datasets\segmentation\images\train','*.jpg']);
for i=1:length(ims)
    I = imread(['C:\Users\rachi\Downloads\Seg_Dataset\coffee-datasets\segmentation\images\train',ims(i).name]);
    id = str2num(ims(i).name(1:end-4));
    severity = dataset{id,7};
    if severity==0
        imwrite(I,['C:\Users\rachi\Downloads\Severity_Images\0\',ims(i).name(1:end-4),'.jpg']);
    elseif severity==1
        imwrite(I,['C:\Users\rachi\Downloads\Severity_Images\1\',ims(i).name(1:end-4),'.jpg']);
    elseif severity==2
        imwrite(I,['C:\Users\rachi\Downloads\Severity_Images\2\',ims(i).name(1:end-4),'.jpg']);
    elseif severity==3
        imwrite(I,['C:\Users\rachi\Downloads\Severity_Images\3\',ims(i).name(1:end-4),'.jpg']);
    elseif severity==4
        imwrite(I,['C:\Users\rachi\Downloads\Severity_Images\4\',ims(i).name(1:end-4),'.jpg']);
    end
    i;
end