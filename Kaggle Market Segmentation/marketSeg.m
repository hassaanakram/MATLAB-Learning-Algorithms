%% Analysing the kaggle dataset for market segmentation

%% CLEARING THE WORKSPACE
clc; close all; clear all;

%% IMPORTING DATA FROM THE CSV FILE
X = importfile('E:\Machine  Learning\Week-8\Mall_Customers.csv', 2, 201);
m = size(X, 1);
n = size(X, 2);

%% SETTING UP THE PARAMETRES FOR K-MEANS
numClusters = 5; %Initially going with 5 clusters
clusterCentroids = zeros(numClusters, n); %A k * n matrix of centroids
assignedClusters = cell(numClusters, 1);

%% RANDOM INITIALIZATION OF clusterCentroids
sel = randperm(m, numClusters);
clusterCentroids = X(sel, :);

%% STANDARD DEVIATION BASED INITIALIZATION OF clusterCentroids -- FAILED
% stdDev = std(X);
% stdVect = [stdDev; 2*stdDev; 3*stdDev; 4*stdDev; 5*stdDev];
% clusterCentroids = stdVect;
%% CALCULATION AND MINIMIZATION OF COST
iterations = 1;
maxIterations = 500;
tempNorm = zeros(1, numClusters);
norms = zeros(m, 1);
means = zeros(numClusters, n);
prevCost = Inf;
newCost = Inf;
%cost = zeros(1,maxIterations);

%THIS LOOP WILL RUN TILL CONVERGENCE
figure;
hold on
while(1)
    %ITERATING OVER THE TRAINING EXAMPLES TO ASSIGN THEM TO CLUSTERS
    for i = 1 : m,
        for j = 1 : numClusters,
            dist = X(i,:) - clusterCentroids(j, :);
            tempNorm(j) = dist * dist';
        end;
        [norms(i), tempAssgCluster] = min(tempNorm);
        assignedClusters{tempAssgCluster} = horzcat(assignedClusters...
                                           {tempAssgCluster}, [i]);
    end;
    %COST ESTIMATION
    prevCost = newCost;
    newCost = calculateCost(X, norms);
    cost(iterations) = newCost;
    %Checking the convergence condition
    if((prevCost - newCost <= 0.001) || (iterations == maxIterations))
        break;
    end

    %Moving the centroids
    for i = 1 : numClusters,
        clusterPoints = X(assignedClusters{i}, :);
        means(i,:) = 1/(size(clusterPoints,1)) .* sum(clusterPoints);
    end;
    clusterCentroids = means;

%Plotting iterations-cost curve
plot(1:iterations, cost, 'k','LineWidth',1.5);
iterations = iterations + 1
assignedClusters = cell(numClusters, 1);
end;
%% ANALYSIS
fprintf('Loop completed. Press any key to continue\n');
pause;
fprintf('Iterations: %d\nCost Difference: %f\nFinal Cost: %f\n', iterations, ...
        prevCost-newCost, cost(iterations));
%Plotting data points vs different parametres
colors = ['r','g','b','k','y','m','c'];
figure;
hold on
for i = 1 : numClusters,
    clusterPoints = X(assignedClusters{i}, :);
    scatter(clusterPoints(:,3), clusterPoints(:,4), colors(i));
    scatter(clusterCentroids(i,3), clusterCentroids(i,4), colors(i), 'd', ...
                                'LineWidth', 5);

end;
hold off

figure;
hold on
for i = 1 : numClusters,
    clusterPoints = X(assignedClusters{i}, :);
    scatter(clusterPoints(:,2), clusterPoints(:,4), colors(i));
    scatter(clusterCentroids(i,2), clusterCentroids(i,4), colors(i), 'd', ...
                                'LineWidth', 5);
end;
hold off










