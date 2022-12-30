ims = dir(['C:\Users\rachi\Downloads\Severity_new_500\severity_of_500_images_and_its_mask\mask\1\','*.png']);
data = [];
for i=2%:length(ims)
    I = imread(['C:\Users\rachi\Downloads\Severity_new_500\severity_of_500_images_and_its_mask\mask\1\',ims(i).name]);
    a1 = I(:,:,1);
    a2 = I(:,:,2);
    a3 = I(:,:,3);

    v=cell2mat(arrayfun(@(x1,x2,x3) [x1 x2 x3],a1(:),a2(:),a3(:),'un',0));
    [a,b,c]=unique(v,'rows','stable');
    idx=histc(c,(1:size(a,1))');
    if length(idx)==3
        index_1 = find(ismember(a, [255 0 0],'rows'));
        index_2 = find(ismember(a, [0 176 0],'rows'));
        metric = idx(index_1)/((idx(index_1)+idx(index_2)));
    else 
        metric = 0;
    end
    temp = [i ; metric];
    data = [data temp];
    i
end

