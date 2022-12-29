%NSS quality aware feature extraction
setDir = fullfile('BUSI');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
net = efficientnetb0
inputSize = net.Layers(1).InputSize;
%analyzeNetwork(net)

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

model = fitniqe(imds);

features_train = []; features_test=[];
for i = 1:length(augimdsTrain.Files)
    I = imread(augimdsTrain.Files{i,1});
    temp5=niqe(I,model);
    I = rgb2gray(imresize(I,[128,128]));
    temp1=extractHOGFeatures(I,'CellSize',[64 64],'BlockSize',[2 2],'NumBins',8);
    temp2=extractHOGFeatures(I,'CellSize',[32 32],'BlockSize',[2 2],'NumBins',8);
    temp3=extractHOGFeatures(I,'CellSize',[16 16],'BlockSize',[2 2],'NumBins',8);
    temp4=extractLBPFeatures(I,'Upright',false);
    temp = [double(temp1) double(temp2) double(temp3) double(temp4) double(temp5)];
    features_train = [features_train ; temp];
end
for i = 1:length(augimdsTest.Files)
    I = imread(augimdsTest.Files{i,1});
    temp5=niqe(I,model);
    I = rgb2gray(imresize(I,[128,128]));
    temp1=extractHOGFeatures(I,'CellSize',[64 64],'BlockSize',[2 2],'NumBins',8);
    temp2=extractHOGFeatures(I,'CellSize',[32 32],'BlockSize',[2 2],'NumBins',8);
    temp3=extractHOGFeatures(I,'CellSize',[16 16],'BlockSize',[2 2],'NumBins',8);
    temp4=extractLBPFeatures(I,'Upright',false);
    temp = [double(temp1) double(temp2) double(temp3) double(temp4) double(temp5)];
    features_test = [features_test ; temp];
end
classifier = fitcecoc(features_train,YTrain);
YPred = predict(classifier,features_test);
accuracy = sum(YTest==YPred)/size(YTest,1) 
C = confusionmat(YTest,YPred)

Y = tsne([features_train ; features_test]);
figure,
gscatter(Y(:,1),Y(:,2),[YTrain;YTest]);
grid on
grid minor



%%
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end



layer = 'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
whos featuresTrain
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
accuracy = sum(YTest==YPred)/size(YTest,1) 
C = confusionmat(YTest,YPred)

% idx = [1 5 10 15];
% figure
% for i = 1:numel(idx)
%     subplot(2,2,i)
%     I = readimage(imdsTest,idx(i));
%     label = YPred(idx(i));
%     imshow(I)
%     title(char(label))
% end

Y = tsne([featuresTrain ; featuresTest]);
figure,
gscatter(Y(:,1),Y(:,2),[YTrain;YTest]);
grid on
grid minor

train = [features_train featuresTrain];
test  = [features_test featuresTest];
classifier = fitcecoc(train,YTrain);
YPred = predict(classifier,test);
accuracy = sum(YTest==YPred)/size(YTest,1) 
C = confusionmat(YTest,YPred)

Y = tsne([featuresTrain ; featuresTest]);
figure,
gscatter(Y(:,1),Y(:,2),[YTrain;YTest]);
grid on
grid minor

%% 
[idx,scores] = fscchi2(train,YTrain);
xlabel('Predictor rank')
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')

train = train';test = test';
train_new = train(idx(1:3000),:);
test_new  = test(idx(1:3000),:);
classifier = fitcecoc(train_new',YTrain);
YPred = predict(classifier,test_new');
accuracy = sum(YTest==YPred)/size(YTest,1) 
C = confusionmat(YTest,YPred)



