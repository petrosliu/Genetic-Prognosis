function [index,erre,e,t]=linSVM(trainingData,testingData,...
                                    selectedGene,estimate)
    num=size(selectedGene,1);
    e=zeros(num,1);
    tic;
    switch estimate
    	case 'resubstitution'
            parfor i=1:num
                X=trainingData.gene(:,selectedGene(i,:));
                y=trainingData.label;
                mdlhat=fitcsvm(X,y,'Cost',[0,0.1;0.1,0]);
                yhat=predict(mdlhat,X);
                e(i)=sum(yhat~=y)/size(y,1);
            end
        case 'leaveout'
            parfor i=1:num
                e(i)=0;
                for j=1:size(trainingData.label,1)
                    X=trainingData.gene(:,selectedGene(i,:));
                    y=trainingData.label;
                    X(j,:)=[];
                    y(j,:)=[];
                    testX=trainingData.gene(j,selectedGene(i,:));
                    testy=trainingData.label(j);
                    mdlhat=fitcsvm(X,y,'Cost',[0,0.1;0.1,0]);
                    yhat=predict(mdlhat,testX);
                    e(i)=e(i)+(yhat~=testy);
                end
                e(i)=e(i)/size(trainingData.label,1);
            end
    end
    [erre,index]=min(e);
    X=trainingData.gene(:,selectedGene(index,:));
    y=trainingData.label;
    mdl=fitcsvm(X,y,'Cost',[0,0.1;0.1,0]);
    testX=testingData.gene(:,selectedGene(index,:));
    testy=testingData.label;
    yhat=predict(mdl,testX);
    e=sum(yhat~=testy)/size(testy,1);
    t=toc;
end