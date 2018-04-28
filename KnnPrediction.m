trainData = csvread('~/Downloads/auto_train.csv',1);
testData = csvread('~/Downloads/auto_test.csv',1);

trainMpgData = trainData(:,3);
testMpgData = testData(:,3);

% Q1 Plot of displacement v/s mpg
close all;
figure
subplot(3,3,1)
scatter(trainData(:,1),trainMpgData);
xlabel('Displacement');
ylabel('MPG');
title('Training Data Plot');

% Q2 Training a linear regression model
simpleLinearModel = fitlm(trainData(:,1),trainMpgData,'linear');
trainPredictedMpgLinear = predict(simpleLinearModel, trainData(:,1));
testPredictedMpgLinear = predict(simpleLinearModel, testData(:,1));
subplot(3,3,2);
plot(simpleLinearModel);
xlabel('Displacement');
ylabel('MPG');
title('Linear Model Plot');

trainErrorLinear = calculateTrainError(trainPredictedMpgLinear, trainMpgData);
testErrorLinear = calculateTestError(testPredictedMpgLinear, testMpgData);

%Q3 Training a plynomial regression model of degree 2, 4 and 6

% Polynomial Regression model for degree 2
polynomial2Model = fitlm(trainData(:,1),trainMpgData,'poly2');
trainPredictedMpgPoly2 = predict(polynomial2Model, trainData(:,1));
testPredictedMpgPoly2 = predict(polynomial2Model, testData(:,1));
trainErrorPoly2 = calculateTrainError(trainPredictedMpgPoly2, trainMpgData);
testErrorPoly2 = calculateTestError(testPredictedMpgPoly2, testMpgData);
subplot(3,3,3);
plot(polynomial2Model);
xlabel('Displacement');
ylabel('MPG');
title('Polynomial Model with degree 2');

% Polynomial Regression model for degree 4
polynomial4Model = fitlm(trainData(:,1),trainMpgData,'poly4');
trainPredictedMpgPoly4 = predict(polynomial4Model, trainData(:,1));
testPredictedMpgPoly4 = predict(polynomial4Model, testData(:,1));
trainErrorPoly4 = calculateTrainError(trainPredictedMpgPoly4, trainMpgData);
testErrorPoly4 = calculateTestError(testPredictedMpgPoly4, testMpgData);
subplot(3,3,4);
plot(polynomial4Model);
xlabel('Displacement');
ylabel('MPG');
title('Polynomial Model with degree 4');

% Polynomial Regression model for degree 6
polynomial6Model = fitlm(trainData(:,1),trainMpgData,'poly6');
trainPredictedMpgPoly6 = predict(polynomial6Model, trainData(:,1));
testPredictedMpgPoly6 = predict(polynomial6Model, testData(:,1));
trainErrorPoly6 = calculateTrainError(trainPredictedMpgPoly6, trainMpgData);
testErrorPoly6 = calculateTestError(testPredictedMpgPoly6, testMpgData);
subplot(3,3,5);
plot(polynomial6Model);
xlabel('Displacement');
ylabel('MPG');
title('Polynomial Model with degree 6');

%Q4 Training a Multiple Linear Regression Model
predictorAttributes = [trainData(:,1),trainData(:,2)];
multipleLinearModel = fitlm(predictorAttributes,trainMpgData,'linear');
predictedMpgMultiple = predict(multipleLinearModel,testData(:,1:2));

testErrorMultiple = calculateTestError(predictedMpgMultiple,testMpgData);

%Q5 Training KNN model
knn1PredictedMpg = predictKnn(trainData, testData, 1, 'Euclidean');
testErrork1 = calculateTestError(knn1PredictedMpg, testMpgData);

knn3PredictedMpg = predictKnn(trainData, testData, 3, 'Euclidean');
testErrork3 = calculateTestError(knn3PredictedMpg, testMpgData);

knn20PredictedMpg = predictKnn(trainData, testData, 20, 'Euclidean');
testErrork20 = calculateTestError(knn20PredictedMpg, testMpgData);

%Q7 Variation in KNN model(using Manhattan Distance to find nearest neighbor)
knn3PredictedMpgManhattan = predictKnn(trainData, testData, 3, 'Manhattan');
testErrork3Manhattan = calculateTestError(knn3PredictedMpgManhattan, testMpgData);


% Function for predicting value using KNN
function predictedMpg = predictKnn(trainData, testData, k, distanceMeasure)
    k3Distances = zeros(1,length(trainData));
    predictedMpg = zeros(1,length(testData));
    for i=1:size(testData)
        for j=1:size(trainData)
            if(distanceMeasure == 'Euclidean')
                distance = (((trainData(j,1)-testData(i,1))^2) + ((trainData(j,2)-testData(i,2))^2))^0.5;
                k3Distances(1,j) = distance;
            end
            if(distanceMeasure == 'Manhattan')
                distance = (abs(trainData(j,1)-testData(i,1))) + (abs(trainData(j,2)-testData(i,2)));
                k3Distances(1,j) = distance;
            end
        end
        [k3Distances, indexes] = sort(k3Distances);
        sum = 0;
        for m=1:k
            sum = sum + trainData(indexes(m),3);
        end
        predictedMpg(1,i) = sum/k;
    end
end

% Function for calculating Training Error
function returnValue = calculateTrainError(predictedData, trainMpgData)
    trainSum = 0;
    for i=1:size(trainMpgData)
        trainSquareDiff = (predictedData(i) - trainMpgData(i))^2;
        trainSum = trainSum + trainSquareDiff;
    end
    returnValue = trainSum/2;

end

% Function for calculating Testing Error
function returnValue = calculateTestError(predictedData, testMpgData)
    testSumMultiple = 0;
    for i=1:size(testMpgData)
    squareDiffMultiple = (predictedData(i) - testMpgData(i))^2;
    testSumMultiple = testSumMultiple + squareDiffMultiple;
    end
    returnValue = testSumMultiple/2;
end
