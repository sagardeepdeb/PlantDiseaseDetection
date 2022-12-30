%NSS quality aware feature extraction
setDir = fullfile('C:\Users\rachi\Downloads\Leaf_dataset_Tomato'); %setting directory
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
net = efficientnetb0; %Pre-trained Efficient Net for feature extraction
inputSize = net.Layers(1).InputSize; %determining the input size require for efficient net


augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

YTrain = imdsTrain.Labels; %train and test lables
YTest = imdsTest.Labels;

model = fitniqe(imds); % Model trained to determine the image quality of an input image
%collecting train features and its corresponding lables 
features_train = []; features_test=[];
for i = 1:length(augimdsTrain.Files)
    I = imread(augimdsTrain.Files{i,1}); % Read individual image from the directory
    temp5=niqe(I,model); % calculating the image quality index
    I = rgb2gray(imresize(I,[128,128])); %Resizing each image for further processing
    %calculating HoG features over various scales [64,64],[32,32],[16,16]
    temp1=extractHOGFeatures(I,'CellSize',[64 64],'BlockSize',[2 2],'NumBins',8); 
    temp2=extractHOGFeatures(I,'CellSize',[32 32],'BlockSize',[2 2],'NumBins',8);
    temp3=extractHOGFeatures(I,'CellSize',[16 16],'BlockSize',[2 2],'NumBins',8);
    temp4=extractLBPFeatures(I,'Upright',false); %Extracting LBP features 
    %??
    temp = [double(temp1) double(temp2) double(temp3) double(temp4) double(temp5)];
    features_train = [features_train ; temp];
    i;
end
%collecting test features and its corresponding lables 
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
    i;
end

%SVM classifier to train on Image quality features
classifier = fitcecoc(features_train,YTrain);
YPred = predict(classifier,features_test);
accuracy_iq = sum(YTest==YPred)/size(YTest,1) 
C = confusionmat(YTest,YPred);
%TSNE Plot
Y = tsne([features_train ; features_test]);
figure,
gscatter(Y(:,1),Y(:,2),[YTrain;YTest]);
grid on
grid minor



%To show few training images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%Name of the layer from where the features are collected 

layer = 'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool';
%test and train features are collected separately as Rows 
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
whos featuresTrain
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
accuracy_deep = sum(YTest==YPred)/size(YTest,1); 
C = confusionmat(YTest,YPred);

%PCA
idx = [1 5 10 15];
 figure
 for i = 1:numel(idx)
     subplot(2,2,i)
     I = readimage(imdsTest,idx(i));
     label = YPred(idx(i));
     imshow(I)
     title(char(label))
end

Y = tsne([featuresTrain ; featuresTest]);
figure,
gscatter(Y(:,1),Y(:,2),[YTrain;YTest]);
grid on
grid minor

%concatenating the deep features along with the image quality features

train = [features_train featuresTrain];
test  = [features_test featuresTest];
classifier = fitcecoc(train,YTrain);
YPred = predict(classifier,test);
accuracy_combine = sum(YTest==YPred)/size(YTest,1); 
C = confusionmat(YTest,YPred);

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
train_new = train(idx(1:2800),:);
test_new  = test(idx(1:2800),:);
classifier = fitcecoc(train_new',YTrain);
YPred = predict(classifier,test_new');
accuracy_selected = sum(YTest==YPred)/size(YTest,1); 
C = confusionchart(YTest,YPred)


 accuracy_selected = [];
 for i=2800:3179
     train_new = train(idx(1:i),:);
     test_new  = test(idx(1:i),:);
     classifier = fitcecoc(train_new',YTrain);
     YPred = predict(classifier,test_new');
     temp = sum(YTest==YPred)/size(YTest,1); 
     accuracy_selected = [accuracy_selected temp];
     i;
 end
 
 figure,plot(2800:3179,accuracy_selected)
 grid on
 grid minor
