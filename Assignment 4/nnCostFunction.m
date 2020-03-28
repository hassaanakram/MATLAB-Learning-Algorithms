function [J grad a2 a3 a4 a5 h] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size,...
                                   hidden_layer3_size,...
                                   hidden_layer4_size,...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1, Theta2...
%Theta3, Theta4, the weight matrices

% for our 6 layer neural network
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

% Setup some useful variables
m = size(X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%Append the bias units
X = [ones(m,1), X];
%First hidden layer
z2 = Theta1*X';
a2 = sigmoid(z2);
%Append the bias unit
a2 = [ones(1, size(a2,2)); a2];
%Second hidden layer
z3 = Theta2 * a2;
a3 = sigmoid(z3);
%Append the bias unit
a3 = [ones(1, size(a3,2)); a3];
%Third hidden layer
z4 = Theta3 * a3;
a4 = sigmoid(z4);
%Append the bias unit
a4 = [ones(1,size(a4,2)); a4];
%Fourth hidden layer
z5 = Theta4 * a4;
a5 = sigmoid(z5);
%Prediction
%Append the bias unit
a5 = [ones(1,size(a5,2)); a5];
z6 = Theta5 * a5;
h = sigmoid(z6);

%Preparing the y-matrix
new_y = zeros(m, num_labels);
for i = 1: m
new_y (i, y(i)) = 1;
end
new_y = new_y';

%Cost calc.
J = trace( 1./m .* (-1.*new_y*log(h)' - (1-new_y)*log(1 - h)'));
regCost = lambda./(2.*m) .* (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:,2:end).^2))...
    + sum(sum(Theta3(:, 2:end).^2))+ sum(sum(Theta4(:, 2:end).^2))+ sum(sum(Theta5(:, 2:end).^2)));
%Adding the regularization factor
J = J + regCost;
%Errors
delta_6 = h - new_y;
delta_5 = ((Theta5(:,2:end))'*delta_6).*sigmoidGradient(z5);
delta_4 = ((Theta4(:,2:end))'*delta_5).*sigmoidGradient(z4);
delta_3 = ((Theta3(:,2:end))'*delta_4).*sigmoidGradient(z3);
delta_2 = ((Theta2(:,2:end))'*delta_3).*sigmoidGradient(z2);

%Gradients
Theta5_grad = 1./m .* (delta_6 * a5') + (lambda/m)*[zeros(size(Theta5,1),1)...
    , Theta5(:,2:end)];
Theta4_grad = 1./m .* (delta_5 * a4') + (lambda/m)*[zeros(size(Theta4,1),1)...
    , Theta4(:,2:end)];
Theta3_grad = 1./m .* (delta_4 * a3') + (lambda/m) *[zeros(size(Theta3,1),1)...
    , Theta3(:,2:end)];
Theta2_grad = 1./m .* (delta_3*a2') + (lambda/m)*[zeros(size(Theta2,1),1)...
    , Theta2(:,2:end)];
Theta1_grad = 1./m .* (delta_2*X) + (lambda/m)*[zeros(size(Theta1,1),1)...
    ,Theta1(:,2:end)];

% Theta2_grad(:, 2 : end) = Theta2_grad(:, 2:end) + (lambda/m) * (Theta2_grad...
%     (:, 2: end));
% Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * (Theta1_grad...
%     (:,2:end));
    
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:) ; Theta3_grad(:); Theta4_grad(:)...
    ; Theta5_grad(:)];

end
