clc; clear all; close all;

%% Load images for training/validation of CNN
matlabpath = ('C:\Users\jan-k\Dropbox\Honza\Škola\Výuka\Moje_Předměty\MaS_zimní_semestr\MaS_2023_2024\Projekt\CNN\Face_database');

data = fullfile(matlabpath);

imds = imageDatastore(data,'IncludeSubfolders', true,'LabelSource','foldernames');

%% Dataset split: 0.7 means: 70% for testing and 30% for validation
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomize');

%% Image count in each class
labelCount = countEachLabel(imds)
labelCount = countEachLabel(imdsTrain)
labelCount = countEachLabel(imdsValidation)

%% Loading pretrained CNN
net=googlenet;

net.Layers(1);
inputSize = net.Layers(1).InputSize;

%% Image augmentation
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation , ...
    'ColorPreprocessing', 'gray2rgb');

%% Transfer learning
numClasses = numel(categories(imdsTrain.Labels))

lgraph = layerGraph(net);

fcLayer = fullyConnectedLayer(numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,'Name', 'FC_Layer');
clsLayer = classificationLayer('Name', 'OutputLayer');
lgraphNew = replaceLayer(lgraph,"loss3-classifier",fcLayer);
lgraphNew = replaceLayer(lgraphNew,"output",clsLayer);
%% CNN hyperparameters settings
options = trainingOptions('sgdm', ...
                          'MiniBatchSize',20, ...
                          'MaxEpoch',20, ...
                          'InitialLearnRate',1e-5, ...
                          'ValidationData',augimdsValidation, ...
                          'Shuffle','every-epoch', ...
                          'ValidationFrequency',10, ...
                          'Verbose',true, ...
                          'Plots','Training-progress') ...
%                           'ExecutionEnvironment','gpu');
                          
%% CNN training 
GoogleNetwork = trainNetwork(augimdsTrain,lgraphNew,options);

%% Evaluation
[YPred] = classify(GoogleNetwork,augimdsValidation);
plotconfusion(imdsValidation.Labels,YPred);
c = confusionmat(imdsValidation.Labels,YPred);

tp = c(1,1);
fp = c(2,1);
tn = c(2,2);
fn = c(1,2);

sensitivity = tp/(tp + fn);  
specificity = tn/(tn + fp);  
precision = tp / (tp + fp);
FPR = fp/(tn+fp);
Accuracy = (tp+tn)./(tp+fp+tn+fn);
recall = tp / (tp + fn);
F1 = (2 * precision * recall) / (precision + recall);
Epochs = options.MaxEpochs;
LearnRate = options.InitialLearnRate;

GoogleNetTab = table(Accuracy, sensitivity, ...
    specificity, precision, FPR, recall, F1, Epochs, LearnRate)

