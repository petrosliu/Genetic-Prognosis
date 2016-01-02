clear ; close all; clc; fclose all;

%% Read Database
fprintf('Reading Database...\n');
tic;
testingData=readdata('Testing_Data.txt');
trainingData=readdata('Training_Data.txt');
[trainNo,geneNo]=size(trainingData.gene);
testNo=size(testingData.gene,1);
fprintf('%.2fs\n\tTraining sample:%d\n\tTesting sample:%d\n\tGene:%d\n'...
    ,toc,trainNo,testNo,geneNo);
outstream=zeros(48,12);

%% Enable 4 cores parallel computing
parpool(4);

%% Resubstitution

%% LDA
%%
%% Exhaustive search 1
fprintf('Diagonal LDA with Exhaustive search 1 ...\n');
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,1,1,1);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 2
fprintf('Diagonal LDA with Exhaustive search 2 ...\n');
selectedGene=nchoosek(1:geneNo,2);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,1,1,2);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 3
fprintf('Diagonal LDA with Exhaustive search 3 ...\n');
selectedGene=nchoosek(1:geneNo,3);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,1,1,3);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Sequential forward search 1
fprintf('Diagonal LDA with Sequential forward search 1 ...\n');
bestGene=zeros(1,5);
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,1,2,1);
outstream(cnt,1:12)=out;
bestGene(1)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 2
fprintf('Diagonal LDA with Sequential forward search 2 ...\n');
selectedGene(:,1)=1:geneNo;
selectedGene(selectedGene==bestGene(1))=[];
selectedGene(:,2)=bestGene(1);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,1,2,2);
outstream(cnt,1:12)=out;
bestGene(1:2)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 3
fprintf('Diagonal LDA with Sequential forward search 3 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2))=[];
selectedGene(:,2:3)=ones(size(selectedGene,1),1)*bestGene(1:2);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,1,2,3);
outstream(cnt,1:12)=out;
bestGene(1:3)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 4
fprintf('Diagonal LDA with Sequential forward search 4 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3))=[];
selectedGene(:,2:4)=ones(size(selectedGene,1),1)*bestGene(1:3);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,1,2,4);
outstream(cnt,1:12)=out;
bestGene(1:4)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 5
fprintf('Diagonal LDA with Sequential forward search 5 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3)...
            |selectedGene==bestGene(4))=[];
selectedGene(:,2:5)=ones(size(selectedGene,1),1)*bestGene(1:4);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,1,2,5);
outstream(cnt,1:12)=out;
bestGene(1:5)=selectedGene(index,:);
clear selectedGene;

%% 3NN
%%
%% Exhaustive search 1
fprintf('3NN with Exhaustive search 1 ...\n');
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,1,1,1);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 2
fprintf('3NN with Exhaustive search 2 ...\n');
selectedGene=nchoosek(1:geneNo,2);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,1,1,2);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 3
fprintf('3NN with Exhaustive search 3 ...\n');
selectedGene=nchoosek(1:geneNo,3);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,1,1,3);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Sequential forward search 1
fprintf('3NN with Sequential forward search 1 ...\n');
bestGene=zeros(1,5);
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,1,2,1);
outstream(cnt,1:12)=out;
bestGene(1)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 2
fprintf('3NN with Sequential forward search 2 ...\n');
selectedGene(:,1)=1:geneNo;
selectedGene(selectedGene==bestGene(1))=[];
selectedGene(:,2)=bestGene(1);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,1,2,2);
outstream(cnt,1:12)=out;
bestGene(1:2)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 3
fprintf('3NN with Sequential forward search 3 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2))=[];
selectedGene(:,2:3)=ones(size(selectedGene,1),1)*bestGene(1:2);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,1,2,3);
outstream(cnt,1:12)=out;
bestGene(1:3)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 4
fprintf('3NN with Sequential forward search 4 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3))=[];
selectedGene(:,2:4)=ones(size(selectedGene,1),1)*bestGene(1:3);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,1,2,4);
outstream(cnt,1:12)=out;
bestGene(1:4)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 5
fprintf('3NN with Sequential forward search 5 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3)...
            |selectedGene==bestGene(4))=[];
selectedGene(:,2:5)=ones(size(selectedGene,1),1)*bestGene(1:4);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,1,2,5);
outstream(cnt,1:12)=out;
bestGene(1:5)=selectedGene(index,:);
clear selectedGene;

%% SVM
%%
%% Exhaustive search 1
fprintf('Linear SVM with Sequential forward search 1 ...\n');
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,1,1,1);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 2
fprintf('Linear SVM with Sequential forward search 2 ...\n');
selectedGene=nchoosek(1:geneNo,2);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,1,1,2);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 3
fprintf('Linear SVM with Sequential forward search 3 ...\n');
selectedGene=nchoosek(1:geneNo,3);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,1,1,3);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Sequential forward search 1
fprintf('Linear SVM with Sequential forward search 1 ...\n');
bestGene=zeros(1,5);
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,1,2,1);
outstream(cnt,1:12)=out;
bestGene(1)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 2
fprintf('Linear SVM with Sequential forward search 2 ...\n');
selectedGene(:,1)=1:geneNo;
selectedGene(selectedGene==bestGene(1))=[];
selectedGene(:,2)=bestGene(1);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,1,2,2);
outstream(cnt,1:12)=out;
bestGene(1:2)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 3
fprintf('Linear SVM with Sequential forward search 3 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2))=[];
selectedGene(:,2:3)=ones(size(selectedGene,1),1)*bestGene(1:2);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,1,2,3);
outstream(cnt,1:12)=out;
bestGene(1:3)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 4
fprintf('Linear SVM with Sequential forward search 4 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3))=[];
selectedGene(:,2:4)=ones(size(selectedGene,1),1)*bestGene(1:3);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,1,2,4);
outstream(cnt,1:12)=out;
bestGene(1:4)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 5
fprintf('Linear SVM with Sequential forward search 5 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3)...
            |selectedGene==bestGene(4))=[];
selectedGene(:,2:5)=ones(size(selectedGene,1),1)*bestGene(1:4);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,...
    'resubstitution');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,1,2,5);outstream(cnt,1:12)=out;
bestGene(1:5)=selectedGene(index,:);
clear selectedGene;

%% Leaveoneout

%% LDA
%%
%% Exhaustive search 1
fprintf('Diagonal LDA with Exhaustive search 1 ...\n');
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,2,1,1);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 2
fprintf('Diagonal LDA with Exhaustive search 2 ...\n');
selectedGene=nchoosek(1:geneNo,2);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,2,1,2);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 3
fprintf('Diagonal LDA with Exhaustive search 3 ...\n');
selectedGene=nchoosek(1:geneNo,3);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,2,1,3);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Sequential forward search 1
fprintf('Diagonal LDA with Sequential forward search 1 ...\n');
bestGene=zeros(1,5);
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,2,2,1);
outstream(cnt,1:12)=out;
bestGene(1)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 2
fprintf('Diagonal LDA with Sequential forward search 2 ...\n');
selectedGene(:,1)=1:geneNo;
selectedGene(selectedGene==bestGene(1))=[];
selectedGene(:,2)=bestGene(1);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,2,2,2);
outstream(cnt,1:12)=out;
bestGene(1:2)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 3
fprintf('Diagonal LDA with Sequential forward search 3 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2))=[];
selectedGene(:,2:3)=ones(size(selectedGene,1),1)*bestGene(1:2);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,2,2,3);
outstream(cnt,1:12)=out;
bestGene(1:3)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 4
fprintf('Diagonal LDA with Sequential forward search 4 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3))=[];
selectedGene(:,2:4)=ones(size(selectedGene,1),1)*bestGene(1:3);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,2,2,4);
outstream(cnt,1:12)=out;
bestGene(1:4)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 5
fprintf('Diagonal LDA with Sequential forward search 5 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3)...
            |selectedGene==bestGene(4))=[];
selectedGene(:,2:5)=ones(size(selectedGene,1),1)*bestGene(1:4);
[index,erre,e,t]=diaLDA(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,1,2,2,5);
outstream(cnt,1:12)=out;
bestGene(1:5)=selectedGene(index,:);
clear selectedGene;

%% 3NN
%%
%% Exhaustive search 1
fprintf('3NN with Exhaustive search 1 ...\n');
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,2,1,1);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 2
fprintf('3NN with Exhaustive search 2 ...\n');
selectedGene=nchoosek(1:geneNo,2);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,2,1,2);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 3
fprintf('3NN with Exhaustive search 3 ...\n');
selectedGene=nchoosek(1:geneNo,3);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,2,1,3);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Sequential forward search 1
fprintf('3NN with Sequential forward search 1 ...\n');
bestGene=zeros(1,5);
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,2,2,1);
outstream(cnt,1:12)=out;
bestGene(1)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 2
fprintf('3NN with Sequential forward search 2 ...\n');
selectedGene(:,1)=1:geneNo;
selectedGene(selectedGene==bestGene(1))=[];
selectedGene(:,2)=bestGene(1);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,2,2,2);
outstream(cnt,1:12)=out;
bestGene(1:2)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 3
fprintf('3NN with Sequential forward search 3 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2))=[];
selectedGene(:,2:3)=ones(size(selectedGene,1),1)*bestGene(1:2);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,2,2,3);
outstream(cnt,1:12)=out;
bestGene(1:3)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 4
fprintf('3NN with Sequential forward search 4 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3))=[];
selectedGene(:,2:4)=ones(size(selectedGene,1),1)*bestGene(1:3);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,2,2,4);
outstream(cnt,1:12)=out;
bestGene(1:4)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 5
fprintf('3NN with Sequential forward search 5 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3)...
            |selectedGene==bestGene(4))=[];
selectedGene(:,2:5)=ones(size(selectedGene,1),1)*bestGene(1:4);
[index,erre,e,t]=threeNN(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,2,2,2,5);
outstream(cnt,1:12)=out;
bestGene(1:5)=selectedGene(index,:);
clear selectedGene;

%% SVM
%%
%% Exhaustive search 1
fprintf('Linear SVM with Sequential forward search 1 ...\n');
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,2,1,1);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 2
fprintf('Linear SVM with Sequential forward search 2 ...\n');
selectedGene=nchoosek(1:geneNo,2);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,2,1,2);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Exhaustive search 3
fprintf('Linear SVM with Sequential forward search 3 ...\n');
selectedGene=nchoosek(1:geneNo,3);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,2,1,3);
outstream(cnt,1:12)=out;
clear selectedGene;

%% Sequential forward search 1
fprintf('Linear SVM with Sequential forward search 1 ...\n');
bestGene=zeros(1,5);
selectedGene(:,1)=1:geneNo;
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,2,2,1);
outstream(cnt,1:12)=out;
bestGene(1)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 2
fprintf('Linear SVM with Sequential forward search 2 ...\n');
selectedGene(:,1)=1:geneNo;
selectedGene(selectedGene==bestGene(1))=[];
selectedGene(:,2)=bestGene(1);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,2,2,2);
outstream(cnt,1:12)=out;
bestGene(1:2)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 3
fprintf('Linear SVM with Sequential forward search 3 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2))=[];
selectedGene(:,2:3)=ones(size(selectedGene,1),1)*bestGene(1:2);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,2,2,3);
outstream(cnt,1:12)=out;
bestGene(1:3)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 4
fprintf('Linear SVM with Sequential forward search 4 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3))=[];
selectedGene(:,2:4)=ones(size(selectedGene,1),1)*bestGene(1:3);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,2,2,4);
outstream(cnt,1:12)=out;
bestGene(1:4)=selectedGene(index,:);
clear selectedGene;

%% Sequential forward search 5
fprintf('Linear SVM with Sequential forward search 5 ...\n');
selectedGene(:,1)=nchoosek(1:geneNo,1);
selectedGene(selectedGene==bestGene(1)...
            |selectedGene==bestGene(2)...
            |selectedGene==bestGene(3)...
            |selectedGene==bestGene(4))=[];
selectedGene(:,2:5)=ones(size(selectedGene,1),1)*bestGene(1:4);
[index,erre,e,t]=linSVM(trainingData,testingData,selectedGene,'leaveout');
[cnt,out]=savedata(selectedGene,index,erre,e,t,3,2,2,5);
outstream(cnt,1:12)=out;
bestGene(1:5)=selectedGene(index,:);
clear selectedGene;

%% End of parallel computing
delete(gcp);

%% Save data
savefile(outstream,trainingData.genename,'gene.mat');