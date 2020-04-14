function [ cost ] = calculateCost( X, norms )
%% THIS FUNCTION CALCULATES AND RETURNS THE COST FOR A GIVEN ARRANGMENT
%OF THE K-MEANS METHOD

%% USEFUL VARIABLES
m = size(X, 1);

%% COST CALCULATION
cost = 1/m * sum(norms);
end

