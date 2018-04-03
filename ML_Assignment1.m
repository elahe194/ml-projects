dataSet = csvread('~/Downloads/glasshw1.csv',0,1);

fprintf('Results for the entire dataset are: \n\n')
[misclassifiedClasses,errorPercentage]=predictClasses(dataSet,dataSet,'gaussian');
fprintf('\nNumber of misclassfied examples are %f \n',misclassifiedClasses)
fprintf('The error is %f percentage.\n\n',errorPercentage)
        
fprintf('*** Results for the 5-Fold cross validation are: *** \n')
performFiveCrossValidation(dataSet,'fiveFold');

fprintf('\n*** Results for the Zero-R 5-Fold cross validation are: *** \n')
performFiveCrossValidation(dataSet,'zeroR');


function [misclassifiedClasses,errorPercentage]= predictClasses(trainDataSet,testDataSet,modelName)
    trainDataSetSize = size(trainDataSet,1);
    testDataSetSize = size(testDataSet,1);

    % Split dataset according to Classes
    glassDataSet = trainDataSet(trainDataSet(:,10) == 1, :);
    nonGlassDataSet = trainDataSet(trainDataSet(:,10) == 2, :);
    
    % Calculating Prior Probabilities
    glassClassProbability = size(glassDataSet,1)/trainDataSetSize;
    nonGlassClassProbability = size(nonGlassDataSet,1)/trainDataSetSize;

    % Calculating Mean
    mean_class1=calculateMean(glassDataSet);
    mean_class2=calculateMean(nonGlassDataSet);

    % Calculating Variance
    variance_class1=calculateVariance(glassDataSet);
    variance_class2=calculateVariance(nonGlassDataSet);
    
    %Printing for Naive Bayes
    if(strcmp(modelName,'gaussian'))
        fprintf('Estimated Prior Probability [P(C)] for each class:\n')
        fprintf('For Class 1 : %f\n',glassClassProbability)
        fprintf('For Class 2 :%f\n\n',nonGlassClassProbability)
        fprintf('Mean for all the attributes of class 1 is :\n') 
        fprintf('%f\n',mean_class1)
        fprintf('\nVariance for all the attributes of class 1 is :\n') 
        fprintf('%f\n',variance_class1)
        fprintf('\nMean for all the attributes of class 2 is :\n') 
        fprintf('%f\n',mean_class2)
        fprintf('\nVariance for all the attributes of class 2 is :\n') 
        fprintf('%f\n',variance_class2)
        fprintf('\n')
        
    end
    
    % Predicting class
    classPrediction=zeros(1,testDataSetSize);
    if strcmp(modelName,'zeroR')
        if glassClassProbability>nonGlassClassProbability
            classPrediction(:)=1;
        else
            classPrediction(:)=2;
        end
    else
        for i=1:testDataSetSize
            rowData = testDataSet(i,1:9);
            glassPosteriorPrediction = calculatePosterior(rowData,mean_class1,variance_class1,glassDataSet,glassClassProbability);
            NonGlassPosteriorPrediction = calculatePosterior(rowData,mean_class2,variance_class2,nonGlassDataSet,nonGlassClassProbability);
            if(glassPosteriorPrediction>NonGlassPosteriorPrediction)
                classPrediction(i)=1;
            else
                classPrediction(i)=2;
            end
        end
        fprintf('Predicted Class: \n')
        for i=1:testDataSetSize
            fprintf('Example %d : %d\n',i,classPrediction(i))
        end
    end
   
    % Calculating Error Percentage
    [misclassifiedClasses,errorPercentage]=calculateError(classPrediction,testDataSet);
end

function returnValue= calculatePosterior(rowData,meanVector,varianceVector,classData,priorProb)
    for i=1:(size(classData,2)-1)
        pdfFirstPart=(sqrt(2*pi*varianceVector(i)));
        pdfSecondPart=exp(-1*((rowData(i)-meanVector(i))^2)/(2*varianceVector(i)));
        pdf(i)=pdfSecondPart/pdfFirstPart;
    end
    returnValue = log(priorProb)+sum(log(pdf));
end

function [counter,result] = calculateError(prediction,data)
counter=0;
    for i=1:size(prediction,2)
        %{
        Printing Predictions for the required tuples
        if(i==20 || i==60 || i==100 || i==140 || i==180)
            fprintf('The predicted class for %d tuple is %d \n',i,prediction(1,i));
        end
        %}
        
        %Calculating total misclassified classes
        if(prediction(1,i) ~= data(i,10))
            counter=counter+1;
        end
    end
result=(counter/size(prediction,2)*100);
end

function returnValue = calculateMean(parameterDataSet)
    meanData=zeros(1,length(parameterDataSet(1,1:end-1)));
    for i=1:size(meanData,2)
        meanData(1,i)=(sum(parameterDataSet(:,i)))/size(parameterDataSet,1);
    end
    returnValue=meanData;
end

function returnValue = calculateVariance(parameterDataSet)
    varianceData=zeros(1,length(parameterDataSet(1,1:end-1)));
    for i=1:size(varianceData,2)
        varianceData(1,i)=var(parameterDataSet(:,i));
    end
    returnValue=varianceData;
end

function returnValue = performFiveCrossValidation(parameterDataSet,modelName)
    %Splitting dataset into 5 parts
    crossValidationDataSets = cat(5,parameterDataSet(1:40,:),parameterDataSet(41:80,:),parameterDataSet(81:120,:),parameterDataSet(121:160,:),parameterDataSet(161:200,:));
    totalForAvg=0;
    for i=1:5
        indices = zeros(1,4);
        for j=1:4.
            index=mod((i+j),5);
            if(index==0)
                indices(1,j)=5;
            else
                indices(1,j)=index;
            end
        end
        testData=crossValidationDataSets(:,:,i);
        trainData = vertcat(crossValidationDataSets(:,:,indices(1,1)),crossValidationDataSets(:,:,indices(1,2)),crossValidationDataSets(:,:,indices(1,3)),crossValidationDataSets(:,:,indices(1,4)));
        fprintf('\nResults for fold %d are: \n',i)
        if strcmp(modelName,'fiveFold')
            [misclassifiedClasses,errorPercentage]= predictClasses(trainData,testData,'fiveFold');
        else
            [misclassifiedClasses,errorPercentage]= predictClasses(trainData,testData,'zeroR');
        end
        fprintf('\nNumber of misclassfied examples are %d \n',misclassifiedClasses)
        fprintf('The error is %f percentage.\n',errorPercentage)
        totalForAvg=totalForAvg+errorPercentage;
    end
        fprintf('\nAverage Error : %f \n',totalForAvg/5);
end
