%% MATLAB CNN -Online 
%https://uk.mathworks.com/solutions/deep-learning/examples/training-a-model-from-scratch.html



%%
%Accessing the Data
rawImgDataTrain = uint8 (fread(fid, numImg * numRows * numCols, 'uint8'));

% Reshape the data part into a 4D array
rawImgDataTrain = reshape(rawImgDataTrain, [numRows, numCols, numImgs]);
imgDataTrain(:,:,1,ii) = uint8(rawImgDataTrain(:,:,ii));	
whos imgDataTrain


%%
%Creating and Configuring Network Layers
layers = [
    imageInputLayer([28 28 1])
	
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%%Training the Network
miniBatchSize = 8192;
options = trainingOptions( 'sgdm',...
    'MiniBatchSize', miniBatchSize,...
    'Plots', 'training-progress');

net = trainNetwork(imgDataTrain, labelsTrain, layers, options);	

%%
%Checking Network Accuracy
predLabelsTest = net.classify(imgDataTest);
accuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)
	
%testAccuracy = 0.9913