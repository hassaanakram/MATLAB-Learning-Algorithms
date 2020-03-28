function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%Prediction
h = X * theta;
%Unreg J
Jureg = 1/(2*m) .* sum((h - y).^2);
%Regulrization parametre
regError = (lambda/(2*m)) .* sum(theta(2:end).^2);
%Regularized Cost function
J = Jureg + regError;

%Gradient Calculation
grad(1) =  (1/m).*sum((h - y).*X(:, 1));
for i = 2 : length(grad)
grad(i) = (1/m).*sum((h-y).*X(:, i)) + (lambda/m).*sum(theta(i));
end
% =========================================================================

grad = grad(:);

end
