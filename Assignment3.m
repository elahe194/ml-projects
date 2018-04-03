data = readtable('~/Downloads/sonar.csv');

attributes = data(:,1:end-1);
tempTarget = data(:,end);
target = zeros(1,size(tempTarget,1));
for i=1:size(tempTarget)
    if(tempTarget.(1)(i) == "Mine")
        target(1,i) = 1;
    else
        target(1,i) = 0;
    end
end

m = -2:0.1:3;
y = (16*m.^4) - (32*m.^3) - (8*m.^2) + (10*m) + 9;
plot(m,y);

% Gradient descent for x=-1 and 5 iterations
fprintf("Gradient Descent for x = -1 and 5 iterations \n");
[xValues, fValues] = gradientDescent(5,0.001,-1);
fprintf("First Five Values \n");
printFirstFive(xValues, fValues);
fprintf("\n");

fprintf("Gradient Descent for x = -1 and 1000 iterations\n");
[xValues, fValues] = gradientDescent(1000,0.001,-1);
fprintf("Last Five Values \n");
printLastFive(xValues, fValues);
fprintf("\n");

fprintf("Gradient Descent for x = 2 and 1000 iterations \n");
[xValues, fValues] = gradientDescent(1000,0.001,2);
fprintf("Last Five Values \n");
printLastFive(xValues, fValues);
fprintf("\n");

fprintf("Gradient Descent for x = -1 and eta = 0.01 \n");
[xValues, fValues] = gradientDescent(1000,0.01,-1);
fprintf("First Five Values \n");
printFirstFive(xValues, fValues);
fprintf("\n");
fprintf("Last Five Values \n");
printLastFive(xValues, fValues);
fprintf("\n");

fprintf("Gradient Descent for x = -1 and 100 iterations\n");
[xValues, fValues] = gradientDescent(100,0.05,-1);
fprintf("First Five Values \n");
printFirstFive(xValues, fValues);
fprintf("\n");
fprintf("Last Five Values \n");
printLastFive(xValues, fValues);
fprintf("\n");


weights = ones(1,61);
weights = weights*0.5;

z=0;
error = zeros(1,50);
rAndyDifference = 0;
tempError = 0;
yt = zeros(1,height(attributes));
for k=1:50
    for i=1:height(attributes)
        for j=1:length(weights)-1
            z = z + attributes.(j)(i)*weights(j);
        end
        z = z + weights(1,length(weights));
        z = z * (-1);
        result = 1/(1 + exp(z));
        if(result < exp(-16))
            yt(1,i) = exp(-16);
        else
            yt(1,i) = result;
        end
        tempError = tempError + (target(1,i)*log(yt(1,i)) + (1 - target(1,i))*(log(1-yt(1,i))));
        rAndyDifference = rAndyDifference + (target(1,i)-yt(1,i));
    end
    error(1,k) = tempError * (-1);

    tempSum = 0;
    for i=1:length(weights)-1
        for j=1:height(attributes)
            tempSum = tempSum + (target(1,j)-yt(1,j))*attributes.(i)(j);
        end
        weights(i) = weights(i) + 0.001*tempSum;
    end

    weights(1,length(weights)) = weights(1,length(weights)) + 0.001*rAndyDifference;
end


function [xValues, fValues] = gradientDescent(iterations,eta,x)
    g = @(x) (16*x.^4) - (32*x.^3) - (8*x.^2) + (10*x) + 9;
    derivativeG = @(x) (64*x.^3) - (96*x.^2) - (16*x) + 10;
    xValues = zeros(1,iterations+1);
    fValues = zeros(1,iterations+1);
    xValues(1,1) = x;
    fValues(1,1) = g(x);

    for i=2:iterations+1
        x = x - eta*(derivativeG(x));
        xValues(1,i) = x;
        f = g(x);
        fValues(1,i) = f;
    end
end

function returnValue = printFirstFive(xValues, fValues)
    for i=1:5
        fprintf("Value of x at the start of the iteration %d : %d \n", i, xValues(1,i));
        fprintf("Value of f(x) at the start of the iteration %d : %d \n", i, fValues(1,i));
        
        fprintf("Value of x at the end of the iteration %d : %d \n", i, xValues(1,i+1));
        fprintf("Value of f(x) at the end of the iteration %d : %d \n", i, fValues(1,i+1));
        fprintf("-----------------------End of iteration %d----------------------- \n", i);
    end
end

function returnValue = printLastFive(xValues, fValues)
    for i=length(xValues)-4:length(xValues)
        fprintf("Value of x at the start of the iteration %d : %d \n", i, xValues(1,i-1));
        fprintf("Value of f(x) at the start of the iteration %d : %d \n", i, fValues(1,i-1));
        
        fprintf("Value of x at the end of the iteration %d : %d \n", i, xValues(1,i));
        fprintf("Value of f(x) at the end of the iteration %d : %d \n", i, fValues(1,i));
        fprintf("-----------------------End of iteration %d----------------------- \n", i-1);
    end
end
