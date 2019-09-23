clc; clear;

addpath(genpath('library'));
addpath(genpath('library\nnClass'));
addpath(genpath('matdata'));
addpath(genpath('outputs'));

warning('off','all');

%% Read Dataset and MLP Settings from Parameters File
nnObject=nnClass;

K_fold=3;

nnObject.HIDDEN_ALPHA=0.0001;
nnObject.OUTPUT_ALPHA=0.1;

nnObject.CHOQUET_NODE_SIZE=[4];
nnObject.NETWORK_NODE_SIZE=[3];

nnObject.MAX_EPOCH=1000;
nnObject.epochOutputPeriod=100;
nnObject.ADAP_LEARN_PARAM=[0.1 0.001 100];

%% Read Training, Validation and Test Data and Extract Features
class_truth=nnObject.classTruth();

irisData=load('fisheriris.mat');

features=irisData.meas';
label_unique=unique(irisData.species);
label_groundtruths=zeros(size(label_unique,1),size(features,2));
for i=1:size(label_unique,1)
    label_groundtruths(i,:)=strcmp(irisData.species,label_unique{i});
end
label_groundtruths(label_groundtruths==0)=class_truth(1);
label_groundtruths(label_groundtruths==1)=class_truth(2);
%% Train and Test Data with K-Fold


% Classify with K-fold
main_cv = cvpartition(num2str(label_groundtruths'),'KFold', K_fold);
test_y_all=[]; test_d_all=[];
for j=1:K_fold
    %Get Partitions
    training_x=features(:,main_cv.training(j)); training_d=label_groundtruths(:,main_cv.training(j));
    test_x=features(:,main_cv.test(j)); test_d=label_groundtruths(:,main_cv.test(j));

    %Get Normalization Coefficients and Normalize Partitions 
    meanSig=mean(training_x,2);
    stdSig=std(training_x,0,2);
    nnObject.NORM_COEFF=[meanSig stdSig];

    training_x=nnObject.normalizeX(training_x);
    test_x=nnObject.normalizeX(test_x);
    
    %MLP part
    nnObject.epochOutputFolder=['outputs\TrainingEpoch(Fold ' num2str(j) ')'];
    sub_cv = cvpartition(num2str(training_d'),'HoldOut',0.2);
    validation_x=training_x(:,sub_cv.test); validation_d=training_d(:,sub_cv.test);
    training_x=training_x(:,sub_cv.training); training_d=training_d(:,sub_cv.training);
    
    nnObject = nnObject.backpropagation(training_x,training_d,validation_x,validation_d); %learning weights
    test_y_all = [test_y_all nnObject.choqmlp(test_x)]; %outputs of choqmlp
    
    save(['outputs\trained_mlp_fold' num2str(j) '.mat'],'nnObject');
    test_d_all=[test_d_all test_d];
end

perf_mlp = nnObject.performance(test_d_all,test_y_all);
