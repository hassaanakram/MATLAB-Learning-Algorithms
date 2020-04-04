function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%Values of C and sigma to check hopefully these'll work somehow
C = [0.01,0.1,1.0,10.0];
sigma = [0.01,0.1,1.0,10.0];
%Training the SVM using Training Set over the C and sigma vectors
for i = 1 : length(C),
    for j = 1 : length(sigma),
        model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j))); 
        %Prediction
        pred = svmPredict(model, Xval);
        %Error
        error(i,j) = mean(double(pred ~= yval));

    end;
end;

% %Plots
% figure;
% plot(C, error(:, 1), 'k', 'LineWidth', 7);
% title('C vs Error at constant sigma');
% figure;
% plot(sigma, error(1, :), 'k', 'LineWidth', 7);
% title('Sigma vs Error at a constant C');

[cIndex, sigIndex] = find(error == (min(min(error))));
C = C(cIndex);
sigma = sigma(sigIndex);
% =========================================================================

end
