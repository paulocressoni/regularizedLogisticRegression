function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% number of training examples
m = length(y); 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%% Compute the cost of a particular choice of theta
J = 1/m * sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))) + (lambda / (2*m)) * sum(theta(2:size(theta,1)).^2);

%% Compute the partial derivatives
% theta(1) is not regularized
grad(1) = (1/m) .* sum((sigmoid(X * theta) - y) .* X(:,1));

for i = 2:size(theta,1)
	grad(i) = (1/m) .* sum((sigmoid(X * theta) - y) .* X(:,i)) + (lambda * theta(i)/ m);
end


end
