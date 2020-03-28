%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer1_size = 500;   % 25 hidden units in the first hidden layer
hidden_layer2_size = 500;  %25 hidden units in the second hidden layer
hidden_layer3_size = 500;
hidden_layer4_size = 500;
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');

%Breaking the data to Training, CV and Test sets randomly
sel = randperm(size(X,1));
selCV = sel(1:1000);
selTest = sel(1001:2000);
sel = sel(2001:end);

Xcv = X(selCV, :);
Xtest = X(selTest, :);
X = X(sel, :);
ycv = y(selCV);
ytest = y(selTest);
y = y(sel);
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


% %% ================ Part 2: Setting Parametres ================
% % In this part of the exercise, we load some pre-initialized 
% % neural network parameters.
% 
% fprintf('\nLoading Saved Neural Network Parameters ...\n')
% 
% % Load the weights into variables Theta1 and Theta2
% load('ex4weights.mat');
% 
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, hidden_layer3_size);
initial_Theta4 = randInitializeWeights(hidden_layer3_size, hidden_layer4_size);
initial_Theta5 = randInitializeWeights(hidden_layer4_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:);...
    initial_Theta3(:); initial_Theta4(:); initial_Theta5(:)];


%% ================ Part 3: Compute Cost (Feedforward) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the cost only. You
%  should complete the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, you can verify that
%  your implementation is correct by verifying that you get the same cost
%  as us for the fixed debugging parameters.
%
%  We suggest implementing the feedforward cost *without* regularization
%  first so that it will be easier for you to debug. Later, in part 4, you
%  will get to implement the regularized cost.
%
fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer1_size, ...
                  hidden_layer2_size, hidden_layer3_size,...
                  hidden_layer4_size, num_labels, X, y, lambda);

 fprintf('Cost at Initial NN parameters: %f ',J);
%          '\n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\nChecking Gradients.\n');
checkNNGradients; 

% %% =================== Training NN ===================
fprintf('\nTraining Neural Network... \n')

iterations = 100;
options = optimset('MaxIter', iterations);
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                    input_layer_size, ...
                                    hidden_layer1_size, ...
                                    hidden_layer2_size,...
                                    hidden_layer3_size,...
                                    hidden_layer4_size,...
                                    num_labels, X, y, lambda);
[nn_params, cost, iterations] = fmincg(costFunction, initial_nn_params, options);

% Obtain Thetas back from nn_params
sizeTheta1 = hidden_layer1_size * (input_layer_size+1);
sizeTheta2 = hidden_layer2_size * (hidden_layer1_size+1);
sizeTheta3 = hidden_layer3_size * (hidden_layer2_size+1);
sizeTheta4 = hidden_layer4_size * (hidden_layer3_size + 1);

Theta1 = reshape(nn_params(1:sizeTheta1), ...
                 hidden_layer1_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + sizeTheta1):sizeTheta1+sizeTheta2), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));
Theta3 = reshape(nn_params(sizeTheta1+sizeTheta2+1:sizeTheta1+sizeTheta2+...
    sizeTheta3),hidden_layer3_size,(hidden_layer2_size+1));
Theta4 = reshape(nn_params(sizeTheta1+sizeTheta2+sizeTheta3+1:sizeTheta1+sizeTheta2+...
    sizeTheta3+sizeTheta4),hidden_layer4_size,(hidden_layer3_size+1));
Theta5 = reshape(nn_params(sizeTheta1+sizeTheta2+sizeTheta3+sizeTheta4+1:end)...
    ,num_labels,(hidden_layer4_size+1));

fprintf('Program paused. Press enter to continue.\n');
pause;
% fprintf('Cost vs Iterations: %f\n', cost);
% %Plot Iter-cost graph
% iter = 1 : iterations;
% figure;
%plot(iter, cost, 'b');
% %% ================= Part 9: Visualize Weights =================
% %  You can now "visualize" what the neural network is learning by 
% %  displaying the hidden units to see what features they are capturing in 
% %  the data.
% 
fprintf('\nVisualizing Neural Network... \n')
%Retrieve the learned features
[J grad a2 a3 a4 a5 h] = nnCostFunction(nn_params, input_layer_size, hidden_layer1_size, ...
                  hidden_layer2_size, ...
                  hidden_layer3_size,...
                  hidden_layer4_size,...
                  num_labels, Xcv, ycv, lambda);
% displayData((a2(2:end, :))');
% figure; 
% displayData((a3(2:end,:))');
% figure;
% displayData((a4(2:end,:))');
% figure;
% displayData((a5(2:end, :))');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
% 
% %% ================= Part 10: Implement Predict =================
% %  After training the neural network, we would like to use it to predict
% %  the labels. You will now implement the "predict" function to use the
% %  neural network to predict the labels of the training set. This lets
% %  you compute the training set accuracy.
% 
pred = predict(Theta1, Theta2, Theta3, Theta4, Theta5, Xcv);
% 
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ycv)) * 100);
fprintf('\nCost at the cross valid set %f\n', J);
% 
% 
