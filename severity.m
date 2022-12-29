ims = dir(['D:\Plant Disease Dataset\Rachit\severity of 500 images and its mask\mask\2\','*.png']);
data = [];
for i=1:length(ims)
    I = imread(['D:\Plant Disease Dataset\Rachit\severity of 500 images and its mask\mask\2\',ims(i).name]);
    a1 = I(:,:,1);
    a2 = I(:,:,2);
    a3 = I(:,:,3);

    v=cell2mat(arrayfun(@(x1,x2,x3) [x1 x2 x3],a1(:),a2(:),a3(:),'un',0));
    [a,b,c]=unique(v,'rows','stable');
    idx=histc(c,(1:size(a,1))');
    if length(idx)==3
        metric = min(idx)/sum(idx);
    else 
        metric = 0;
    end
    temp = [i ; metric];
    data = [data temp];
    i
end